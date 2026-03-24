`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : top_nnia
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Top-level integration of the NNIA tiled accelerator.

Connects all modules including buffers, PE array, control logic, and
post-processing stages to execute full neural network inference using
tiled computation.

Key Points
----------
- Integrates full pipeline: input → compute → accumulate → postprocess → output
- Executes tiled matrix computation and assembles final 4x8 output
- Manages tile-level writeback and final output storage
- Aligns hardware flow with Python reference model
- Exposes performance counters for runtime analysis

Role in Pipeline
----------------
- Acts as the system-level orchestrator of NNIA
- Bridges testbench/host inputs with hardware execution
- Produces final inference output and completion status
------------------------------------------------------------------------------
*/

module top_nnia #(
    parameter integer M_TOTAL    = 4,
    parameter integer K_TOTAL    = 16,
    parameter integer N_TOTAL    = 8,
    parameter integer TILE_M     = 4,
    parameter integer TILE_K     = 4,
    parameter integer TILE_N     = 4,
    parameter integer ROWS       = 4,
    parameter integer COLS       = 4,
    parameter integer DATA_WIDTH = 16,
    parameter integer FRAC_BITS  = 8,
    parameter integer ACC_WIDTH  = 40
)(
    input  wire                                           clk,
    input  wire                                           rst,
    input  wire                                           start,

    // Full matrices / vectors from testbench or host-style wrapper
    input  wire signed [(M_TOTAL*K_TOTAL*DATA_WIDTH)-1:0] input_flat,
    input  wire signed [(K_TOTAL*N_TOTAL*DATA_WIDTH)-1:0] weight_flat,
    input  wire signed [(N_TOTAL*DATA_WIDTH)-1:0]         bias_flat,

    // Top-level status
    output wire                                           busy,
    output wire                                           done,
    output wire                                           error,

    // Full final output matrix: row-major [row][n]
    output wire signed [(M_TOTAL*N_TOTAL*DATA_WIDTH)-1:0] final_output_flat,
    output reg                                            output_valid,

    // Passive performance counters
    output wire [31:0]                                    perf_total_cycles,
    output wire [31:0]                                    perf_run_cycles,
    output wire [31:0]                                    perf_run_count,
    output wire [31:0]                                    perf_tile_writes,
    output wire [31:0]                                    perf_psum_saves
);

    // ------------------------------------------------------------------------
    // Safety checks
    // ------------------------------------------------------------------------
    initial begin
        if (M_TOTAL != 4) begin
            $error("top_nnia: Locked NNIA version expects M_TOTAL=4.");
            $finish;
        end

        if (K_TOTAL != 16) begin
            $error("top_nnia: Locked NNIA version expects K_TOTAL=16.");
            $finish;
        end

        if (N_TOTAL != 8) begin
            $error("top_nnia: Locked NNIA version expects N_TOTAL=8.");
            $finish;
        end

        if (TILE_M != 4 || TILE_K != 4 || TILE_N != 4) begin
            $error("top_nnia: Locked NNIA version expects TILE_M=TILE_K=TILE_N=4.");
            $finish;
        end

        if (ROWS != 4 || COLS != 4) begin
            $error("top_nnia: Locked NNIA version expects ROWS=COLS=4.");
            $finish;
        end
    end

    // ------------------------------------------------------------------------
    // Local parameters
    // ------------------------------------------------------------------------
    localparam integer OUT_FLAT_W = M_TOTAL * N_TOTAL * DATA_WIDTH;

    // ------------------------------------------------------------------------
    // Start-edge detect
    // Used to:
    // - load input_buffer and weight_buffer
    // - clear full output register for a new run
    // ------------------------------------------------------------------------
    reg start_d;
    wire start_rise;

    always @(posedge clk) begin
        if (rst)
            start_d <= 1'b0;
        else
            start_d <= start;
    end

    assign start_rise = start & ~start_d;

    // ------------------------------------------------------------------------
    // Done-edge detect
    // tile_controller may hold done high, so use a pulse for counters
    // ------------------------------------------------------------------------
    reg  done_d;
    wire done_rise;

    always @(posedge clk) begin
        if (rst)
            done_d <= 1'b0;
        else
            done_d <= done;
    end

    assign done_rise = done & ~done_d;

    // ------------------------------------------------------------------------
    // tile_addr_gen wires
    // ------------------------------------------------------------------------
    wire [$clog2(M_TOTAL)-1:0] tile_m_base;
    wire [$clog2(K_TOTAL)-1:0] tile_k_base;
    wire [$clog2(N_TOTAL)-1:0] tile_n_base;

    wire first_k_tile;
    wire last_k_tile;
    wire first_output_tile;
    wire last_output_tile;
    wire addr_active;
    wire addr_done;
    wire addr_cmd_error;

    wire addr_start_run;
    wire addr_advance_k;
    wire addr_advance_out;

    // ------------------------------------------------------------------------
    // tile_controller wires
    // ------------------------------------------------------------------------
    wire inbuf_start;
    wire inbuf_advance;
    wire wbuf_start;
    wire wbuf_advance;

    wire psum_clear;
    wire psum_save;

    wire pe_clear_psum;
    wire pe_load_psum;

    wire outbuf_clear;
    wire outbuf_save;

    wire ctrl_error;

    // ------------------------------------------------------------------------
    // input_buffer wires
    // ------------------------------------------------------------------------
    wire signed [(TILE_M*DATA_WIDTH)-1:0] row_act_in;
    wire        [TILE_M-1:0]              row_act_valid_in;
    wire                                  inbuf_stream_active;
    wire [$clog2(TILE_K)-1:0]             inbuf_stream_step;
    wire                                  inbuf_tile_done;
    wire                                  inbuf_cfg_error;

    // ------------------------------------------------------------------------
    // weight_buffer wires
    // ------------------------------------------------------------------------
    wire signed [(TILE_N*DATA_WIDTH)-1:0] col_weight_in;
    wire        [TILE_N-1:0]              col_weight_valid_in;
    wire                                  wbuf_stream_active;
    wire [$clog2(TILE_K)-1:0]             wbuf_stream_step;
    wire                                  wbuf_tile_done;
    wire                                  wbuf_cfg_error;

    // ------------------------------------------------------------------------
    // PE array wires
    // ------------------------------------------------------------------------
    wire signed [(ROWS*DATA_WIDTH)-1:0]     right_edge_act_out;
    wire signed [(COLS*DATA_WIDTH)-1:0]     bottom_edge_weight_out;
    wire        [ROWS-1:0]                  right_edge_act_valid_out;
    wire        [COLS-1:0]                  bottom_edge_weight_valid_out;
    wire signed [(ROWS*COLS*ACC_WIDTH)-1:0] pe_psum_out_flat;

    // ------------------------------------------------------------------------
    // psum buffer wires
    // ------------------------------------------------------------------------
    wire signed [(ROWS*COLS*ACC_WIDTH)-1:0] psum_seed_flat;
    wire signed [(ROWS*COLS*ACC_WIDTH)-1:0] psum_buf_out_flat;
    wire                                    psum_buf_valid;

    // ------------------------------------------------------------------------
    // Post-process wires
    // ------------------------------------------------------------------------
    reg  signed [(COLS*DATA_WIDTH)-1:0]      col_bias_in;
    wire signed [(ROWS*COLS*DATA_WIDTH)-1:0] processed_out_tile_flat;

    // ------------------------------------------------------------------------
    // output_buffer wires
    // ------------------------------------------------------------------------
    wire signed [(ROWS*COLS*DATA_WIDTH)-1:0] outbuf_tile_flat;
    wire                                     outbuf_valid;

    // ------------------------------------------------------------------------
    // Performance counter wires
    // ------------------------------------------------------------------------
    wire [31:0] perf_total_cycles_w;
    wire [31:0] perf_run_cycles_w;
    wire [31:0] perf_run_count_w;
    wire [31:0] perf_tile_writes_w;
    wire [31:0] perf_psum_saves_w;

    // ------------------------------------------------------------------------
    // Delayed tile-writeback context
    // We save tile context on outbuf_save, then one cycle later use the
    // registered output_buffer contents for final 4x8 assembly.
    // ------------------------------------------------------------------------
    reg                                      outbuf_save_d;
    reg [$clog2(N_TOTAL)-1:0]                saved_tile_n_base_d;
    reg                                      saved_last_output_tile_d;

    // ------------------------------------------------------------------------
    // Full final output storage (4x8)
    // ------------------------------------------------------------------------
    reg signed [OUT_FLAT_W-1:0] final_output_store_flat;
    assign final_output_flat = final_output_store_flat;

    // ------------------------------------------------------------------------
    // Load full matrices once at start edge
    // ------------------------------------------------------------------------
    wire input_load_en;
    wire weight_load_en;

    assign input_load_en  = start_rise;
    assign weight_load_en = start_rise;

    // ------------------------------------------------------------------------
    // Bias tile selection
    // col_bias_in[c] = bias_flat[tile_n_base + c]
    // ------------------------------------------------------------------------
    integer bc;
    always @(*) begin
        col_bias_in = {(COLS*DATA_WIDTH){1'b0}};
        for (bc = 0; bc < COLS; bc = bc + 1) begin
            col_bias_in[(bc*DATA_WIDTH) +: DATA_WIDTH] =
                bias_flat[((tile_n_base + bc)*DATA_WIDTH) +: DATA_WIDTH];
        end
    end

    // ------------------------------------------------------------------------
    // tile_addr_gen
    // ------------------------------------------------------------------------
    tile_addr_gen #(
        .M_TOTAL(M_TOTAL),
        .K_TOTAL(K_TOTAL),
        .N_TOTAL(N_TOTAL),
        .TILE_M (TILE_M),
        .TILE_K (TILE_K),
        .TILE_N (TILE_N)
    ) u_tile_addr_gen (
        .clk                 (clk),
        .rst                 (rst),
        .start_run           (addr_start_run),
        .advance_k_tile      (addr_advance_k),
        .advance_output_tile (addr_advance_out),
        .tile_m_base         (tile_m_base),
        .tile_k_base         (tile_k_base),
        .tile_n_base         (tile_n_base),
        .first_k_tile        (first_k_tile),
        .last_k_tile         (last_k_tile),
        .first_output_tile   (first_output_tile),
        .last_output_tile    (last_output_tile),
        .active              (addr_active),
        .all_done            (addr_done),
        .cmd_error           (addr_cmd_error)
    );

    // ------------------------------------------------------------------------
    // tile_controller
    // FIX:
    // use +1 extra drain cycle so final registered PE/MAC result is visible
    // before psum_save
    // ------------------------------------------------------------------------
    tile_controller #(
        .TILE_K       (TILE_K),
        .ROWS         (ROWS),
        .COLS         (COLS),
        .DRAIN_CYCLES ((ROWS - 1) + (COLS - 1) + 1)
    ) u_tile_controller (
        .clk                 (clk),
        .rst                 (rst),
        .start               (start),
        .busy                (busy),
        .done                (done),

        .addr_start_run      (addr_start_run),
        .addr_advance_k      (addr_advance_k),
        .addr_advance_out    (addr_advance_out),

        .first_k_tile        (first_k_tile),
        .last_k_tile         (last_k_tile),
        .first_output_tile   (first_output_tile),
        .last_output_tile    (last_output_tile),
        .addr_active         (addr_active),
        .addr_done           (addr_done),
        .cmd_error           (addr_cmd_error),

        .inbuf_start         (inbuf_start),
        .inbuf_advance       (inbuf_advance),
        .inbuf_tile_done     (inbuf_tile_done),
        .inbuf_stream_active (inbuf_stream_active),
        .inbuf_cfg_error     (inbuf_cfg_error),

        .wbuf_start          (wbuf_start),
        .wbuf_advance        (wbuf_advance),
        .wbuf_tile_done      (wbuf_tile_done),
        .wbuf_stream_active  (wbuf_stream_active),
        .wbuf_cfg_error      (wbuf_cfg_error),

        .psum_clear          (psum_clear),
        .psum_save           (psum_save),

        .pe_clear_psum       (pe_clear_psum),
        .pe_load_psum        (pe_load_psum),

        .outbuf_clear        (outbuf_clear),
        .outbuf_save         (outbuf_save),

        .ctrl_error          (ctrl_error)
    );

    // Stronger top-level error visibility
    assign error = ctrl_error | addr_cmd_error | inbuf_cfg_error | wbuf_cfg_error;

    // ------------------------------------------------------------------------
    // input_buffer
    // ------------------------------------------------------------------------
    input_buffer #(
        .M_ROWS     (M_TOTAL),
        .K_TOTAL    (K_TOTAL),
        .TILE_M     (TILE_M),
        .TILE_K     (TILE_K),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_input_buffer (
        .clk              (clk),
        .rst              (rst),
        .load_en          (input_load_en),
        .input_flat       (input_flat),
        .tile_m_base      (tile_m_base),
        .tile_k_base      (tile_k_base),
        .start_tile       (inbuf_start),
        .advance          (inbuf_advance),
        .row_act_in       (row_act_in),
        .row_act_valid_in (row_act_valid_in),
        .stream_active    (inbuf_stream_active),
        .stream_step      (inbuf_stream_step),
        .tile_done        (inbuf_tile_done),
        .cfg_error        (inbuf_cfg_error)
    );

    // ------------------------------------------------------------------------
    // weight_buffer
    // ------------------------------------------------------------------------
    weight_buffer #(
        .K_TOTAL    (K_TOTAL),
        .N_TOTAL    (N_TOTAL),
        .TILE_K     (TILE_K),
        .TILE_N     (TILE_N),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_weight_buffer (
        .clk                 (clk),
        .rst                 (rst),
        .load_en             (weight_load_en),
        .weight_flat         (weight_flat),
        .tile_k_base         (tile_k_base),
        .tile_n_base         (tile_n_base),
        .start_tile          (wbuf_start),
        .advance             (wbuf_advance),
        .col_weight_in       (col_weight_in),
        .col_weight_valid_in (col_weight_valid_in),
        .stream_active       (wbuf_stream_active),
        .stream_step         (wbuf_stream_step),
        .tile_done           (wbuf_tile_done),
        .cfg_error           (wbuf_cfg_error)
    );

    // ------------------------------------------------------------------------
    // pe_array_4x4
    // ------------------------------------------------------------------------
    pe_array_4x4 #(
        .ROWS       (ROWS),
        .COLS       (COLS),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_pe_array_4x4 (
        .clk                          (clk),
        .rst                          (rst),
        .row_act_in                   (row_act_in),
        .col_weight_in                (col_weight_in),
        .row_act_valid_in             (row_act_valid_in),
        .col_weight_valid_in          (col_weight_valid_in),
        .clear_psum                   (pe_clear_psum),
        .load_psum                    (pe_load_psum),
        .psum_seed_flat               (psum_seed_flat),
        .right_edge_act_out           (right_edge_act_out),
        .bottom_edge_weight_out       (bottom_edge_weight_out),
        .right_edge_act_valid_out     (right_edge_act_valid_out),
        .bottom_edge_weight_valid_out (bottom_edge_weight_valid_out),
        .psum_out_flat                (pe_psum_out_flat)
    );

    // ------------------------------------------------------------------------
    // psum_buffer
    // ------------------------------------------------------------------------
    psum_buffer #(
        .ROWS      (ROWS),
        .COLS      (COLS),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_psum_buffer (
        .clk            (clk),
        .rst            (rst),
        .clear_buf      (psum_clear),
        .save_en        (psum_save),
        .psum_in_flat   (pe_psum_out_flat),
        .psum_seed_flat (psum_seed_flat),
        .psum_out_flat  (psum_buf_out_flat),
        .buf_valid      (psum_buf_valid)
    );

    // ------------------------------------------------------------------------
    // postprocess_array
    // ------------------------------------------------------------------------
    postprocess_array #(
        .ROWS       (ROWS),
        .COLS       (COLS),
        .DATA_WIDTH (DATA_WIDTH),
        .FRAC_BITS  (FRAC_BITS),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_postprocess_array (
        .raw_acc_tile_flat (psum_buf_out_flat),
        .col_bias_in       (col_bias_in),
        .out_tile_flat     (processed_out_tile_flat)
    );

    // ------------------------------------------------------------------------
    // output_buffer
    // ------------------------------------------------------------------------
    output_buffer #(
        .ROWS       (ROWS),
        .COLS       (COLS),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_output_buffer (
        .clk              (clk),
        .rst              (rst),
        .clear_buf        (outbuf_clear),
        .save_en          (outbuf_save),
        .out_tile_in_flat (processed_out_tile_flat),
        .out_tile_flat    (outbuf_tile_flat),
        .buf_valid        (outbuf_valid)
    );

    // ------------------------------------------------------------------------
    // Delay output-buffer save event by one cycle so top writes back the
    // registered output_buffer contents, not the combinational postprocess tile.
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            outbuf_save_d            <= 1'b0;
            saved_tile_n_base_d      <= {($clog2(N_TOTAL)){1'b0}};
            saved_last_output_tile_d <= 1'b0;
        end
        else begin
            outbuf_save_d <= outbuf_save;

            if (outbuf_save) begin
                saved_tile_n_base_d      <= tile_n_base;
                saved_last_output_tile_d <= last_output_tile;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Full output writeback / assembly
    // Capture each registered 4x4 output tile into final 4x8 output storage.
    // Writeback happens one cycle after outbuf_save, using:
    // - outbuf_tile_flat        : registered output tile
    // - saved_tile_n_base_d     : delayed tile context
    // ------------------------------------------------------------------------
    integer wr_r, wr_c;
    integer dst_index, src_index;

    always @(posedge clk) begin
        if (rst) begin
            final_output_store_flat <= {OUT_FLAT_W{1'b0}};
            output_valid            <= 1'b0;
        end
        else begin
            if (start_rise) begin
                final_output_store_flat <= {OUT_FLAT_W{1'b0}};
                output_valid            <= 1'b0;
            end
            else begin
                if (outbuf_save_d) begin
                    for (wr_r = 0; wr_r < ROWS; wr_r = wr_r + 1) begin
                        for (wr_c = 0; wr_c < COLS; wr_c = wr_c + 1) begin
                            dst_index = (wr_r * N_TOTAL) + (saved_tile_n_base_d + wr_c);
                            src_index = (wr_r * COLS) + wr_c;

                            final_output_store_flat[(dst_index*DATA_WIDTH) +: DATA_WIDTH] <=
                                outbuf_tile_flat[(src_index*DATA_WIDTH) +: DATA_WIDTH];
                        end
                    end
                end

                if (outbuf_save_d && saved_last_output_tile_d)
                    output_valid <= 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------------
    // NNIA performance counters
    // Passive observer only
    // ------------------------------------------------------------------------
    nnia_perf_counters #(
        .COUNT_WIDTH(32)
    ) u_nnia_perf_counters (
        .clk              (clk),
        .rst              (rst),
        .start_pulse      (start_rise),
        .done_pulse       (done_rise),
        .tile_write_pulse (outbuf_save_d),
        .psum_save_pulse  (psum_save),

        .total_cycles     (perf_total_cycles_w),
        .run_cycles       (perf_run_cycles_w),
        .run_count        (perf_run_count_w),
        .tile_writes      (perf_tile_writes_w),
        .psum_saves       (perf_psum_saves_w)
    );

    assign perf_total_cycles = perf_total_cycles_w;
    assign perf_run_cycles   = perf_run_cycles_w;
    assign perf_run_count    = perf_run_count_w;
    assign perf_tile_writes  = perf_tile_writes_w;
    assign perf_psum_saves   = perf_psum_saves_w;

endmodule