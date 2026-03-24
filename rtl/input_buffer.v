`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : input_buffer
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Input activation buffer for tiled NNIA execution.

This module stores the full input activation matrix, selects the requested
tile window, and streams one activation vector per cycle into the left edge
of the PE array. A row-wise skew pipeline is applied so activations reach
deeper processing elements with the correct timing alignment.

Skew Scheme
-----------
- Row 0 : 0-cycle delay
- Row 1 : 1-cycle delay
- Row 2 : 2-cycle delay
- Row 3 : 3-cycle delay

Key Behavior
------------
- Stores the flattened input activation matrix
- Latches tile base indices at tile start
- Validates tile configuration before streaming
- Injects one K-step activation slice per cycle
- Applies fixed row-wise skew through internal pipelines
- Generates row_act_in and row_act_valid_in for the PE array
- Reports tile completion and configuration errors

Notes
-----
- This version is locked for the 4x4 tiled NNIA configuration
- Activations are streamed for left-to-right propagation in the PE array
- Skew alignment is required so matching activation and weight values meet
  at the correct processing elements
------------------------------------------------------------------------------
*/

module input_buffer #(
    parameter integer M_ROWS     = 4,
    parameter integer K_TOTAL    = 16,
    parameter integer TILE_M     = 4,
    parameter integer TILE_K     = 4,
    parameter integer DATA_WIDTH = 16
)(
    input  wire                                           clk,
    input  wire                                           rst,

    input  wire                                           load_en,
    input  wire signed [(M_ROWS*K_TOTAL*DATA_WIDTH)-1:0]  input_flat,

    input  wire [$clog2(M_ROWS)-1:0]                      tile_m_base,
    input  wire [$clog2(K_TOTAL)-1:0]                     tile_k_base,
    input  wire                                           start_tile,
    input  wire                                           advance,

    output reg  signed [(TILE_M*DATA_WIDTH)-1:0]          row_act_in,
    output reg         [TILE_M-1:0]                       row_act_valid_in,

    output reg                                            stream_active,
    output reg  [$clog2(TILE_K)-1:0]                      stream_step,
    output reg                                            tile_done,
    output reg                                            cfg_error
);

    initial begin
        if (M_ROWS != 4) begin
            $error("input_buffer: Locked NNIA version expects M_ROWS=4.");
            $finish;
        end

        if (K_TOTAL != 16) begin
            $error("input_buffer: Locked NNIA version expects K_TOTAL=16.");
            $finish;
        end

        if (TILE_M != 4) begin
            $error("input_buffer: Locked NNIA version expects TILE_M=4.");
            $finish;
        end

        if (TILE_K != 4) begin
            $error("input_buffer: Locked NNIA version expects TILE_K=4.");
            $finish;
        end
    end

    // ------------------------------------------------------------------------
    // Full activation storage
    // ------------------------------------------------------------------------
    reg signed [(M_ROWS*K_TOTAL*DATA_WIDTH)-1:0] input_store_flat;

    // Latched tile bases
    reg [$clog2(M_ROWS)-1:0]  tile_m_base_reg;
    reg [$clog2(K_TOTAL)-1:0] tile_k_base_reg;

    // Config validity
    wire tile_m_base_ok;
    wire tile_k_base_ok;
    wire start_cfg_ok;

    assign tile_m_base_ok = (tile_m_base + TILE_M) <= M_ROWS;
    assign tile_k_base_ok = ((tile_k_base + TILE_K) <= K_TOTAL) &&
                            ((tile_k_base % TILE_K) == 0);
    assign start_cfg_ok   = tile_m_base_ok && tile_k_base_ok;

    // ------------------------------------------------------------------------
    // Per-row injected data before skew
    // ------------------------------------------------------------------------
    reg  signed [DATA_WIDTH-1:0] inject_data [0:TILE_M-1];
    reg                          inject_valid[0:TILE_M-1];

    // Row skew pipelines:
    // row r uses stage[r] as its output
    reg signed [DATA_WIDTH-1:0] act_pipe_data [0:TILE_M-1][0:TILE_M-1];
    reg                         act_pipe_valid[0:TILE_M-1][0:TILE_M-1];

    integer r, s;
    integer flat_index;

    // ------------------------------------------------------------------------
    // Combinational read of current tile element to inject into stage0
    // ------------------------------------------------------------------------
    always @(*) begin
        for (r = 0; r < TILE_M; r = r + 1) begin
            inject_data[r]  = {DATA_WIDTH{1'b0}};
            inject_valid[r] = 1'b0;

            if (stream_active) begin
                flat_index = ((tile_m_base_reg + r) * K_TOTAL) +
                             (tile_k_base_reg + stream_step);

                inject_data[r]  = input_store_flat[(flat_index*DATA_WIDTH) +: DATA_WIDTH];
                inject_valid[r] = 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Drive left-edge outputs from skew pipeline taps
    // row r takes stage r
    // ------------------------------------------------------------------------
    always @(*) begin
        row_act_in       = {(TILE_M*DATA_WIDTH){1'b0}};
        row_act_valid_in = {TILE_M{1'b0}};

        for (r = 0; r < TILE_M; r = r + 1) begin
            row_act_in[(r*DATA_WIDTH) +: DATA_WIDTH] = act_pipe_data[r][r];
            row_act_valid_in[r]                      = act_pipe_valid[r][r];
        end
    end

    // ------------------------------------------------------------------------
    // Sequential control + skew pipeline shift
    // Pipeline shifts every cycle so skewed data can flush during controller
    // drain cycles after the last injected stream step.
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            input_store_flat <= {(M_ROWS*K_TOTAL*DATA_WIDTH){1'b0}};
            tile_m_base_reg  <= {($clog2(M_ROWS)){1'b0}};
            tile_k_base_reg  <= {($clog2(K_TOTAL)){1'b0}};
            stream_active    <= 1'b0;
            stream_step      <= {($clog2(TILE_K)){1'b0}};
            tile_done        <= 1'b0;
            cfg_error        <= 1'b0;

            for (r = 0; r < TILE_M; r = r + 1) begin
                for (s = 0; s < TILE_M; s = s + 1) begin
                    act_pipe_data[r][s]  <= {DATA_WIDTH{1'b0}};
                    act_pipe_valid[r][s] <= 1'b0;
                end
            end
        end
        else begin
            tile_done <= 1'b0;
            cfg_error <= 1'b0;

            if (load_en) begin
                input_store_flat <= input_flat;
            end

            // Shift skew pipelines every cycle
            for (r = 0; r < TILE_M; r = r + 1) begin
                for (s = TILE_M-1; s > 0; s = s - 1) begin
                    act_pipe_data[r][s]  <= act_pipe_data[r][s-1];
                    act_pipe_valid[r][s] <= act_pipe_valid[r][s-1];
                end

                act_pipe_data[r][0]  <= inject_data[r];
                act_pipe_valid[r][0] <= inject_valid[r];
            end

            // Start has priority over advance
            if (start_tile) begin
                if (start_cfg_ok) begin
                    tile_m_base_reg <= tile_m_base;
                    tile_k_base_reg <= tile_k_base;
                    stream_active   <= 1'b1;
                    stream_step     <= {($clog2(TILE_K)){1'b0}};
                end
                else begin
                    stream_active   <= 1'b0;
                    stream_step     <= {($clog2(TILE_K)){1'b0}};
                    cfg_error       <= 1'b1;
                end
            end
            else if (stream_active && advance) begin
                if (stream_step == TILE_K-1) begin
                    stream_active <= 1'b0;
                    stream_step   <= {($clog2(TILE_K)){1'b0}};
                    tile_done     <= 1'b1;
                end
                else begin
                    stream_step <= stream_step + 1'b1;
                end
            end
        end
    end

endmodule