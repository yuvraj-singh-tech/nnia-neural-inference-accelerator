`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : weight_buffer
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Weight storage and streaming stage for NNIA tiled execution.

Stores the full weight matrix and streams one tile-aligned weight vector
into the PE array, applying column-wise skew to ensure correct temporal
alignment with activation flow.

Key Points
----------
- Streams weights per K-step into PE array top edge
- Applies column-wise skew for correct PE alignment
- Maintains internal tile configuration and stream control
- tile_done indicates completion of one K-tile stream
- cfg_error flags invalid tile configuration

Role in Pipeline
----------------
- Works with input_buffer for synchronized data injection
- Feeds pe_array_4x4 with aligned weight data
- Supports tiled accumulation across K dimension
------------------------------------------------------------------------------
*/

module weight_buffer #(
    parameter integer K_TOTAL    = 16,
    parameter integer N_TOTAL    = 8,
    parameter integer TILE_K     = 4,
    parameter integer TILE_N     = 4,
    parameter integer DATA_WIDTH = 16
)(
    input  wire                                           clk,
    input  wire                                           rst,

    input  wire                                           load_en,
    input  wire signed [(K_TOTAL*N_TOTAL*DATA_WIDTH)-1:0] weight_flat,

    input  wire [$clog2(K_TOTAL)-1:0]                     tile_k_base,
    input  wire [$clog2(N_TOTAL)-1:0]                     tile_n_base,
    input  wire                                           start_tile,
    input  wire                                           advance,

    output reg  signed [(TILE_N*DATA_WIDTH)-1:0]          col_weight_in,
    output reg         [TILE_N-1:0]                       col_weight_valid_in,

    output reg                                            stream_active,
    output reg  [$clog2(TILE_K)-1:0]                      stream_step,
    output reg                                            tile_done,
    output reg                                            cfg_error
);

    initial begin
        if (K_TOTAL != 16) begin
            $error("weight_buffer: Locked NNIA version expects K_TOTAL=16.");
            $finish;
        end

        if (N_TOTAL != 8) begin
            $error("weight_buffer: Locked NNIA version expects N_TOTAL=8.");
            $finish;
        end

        if (TILE_K != 4) begin
            $error("weight_buffer: Locked NNIA version expects TILE_K=4.");
            $finish;
        end

        if (TILE_N != 4) begin
            $error("weight_buffer: Locked NNIA version expects TILE_N=4.");
            $finish;
        end
    end

    // ------------------------------------------------------------------------
    // Full weight storage
    // ------------------------------------------------------------------------
    reg signed [(K_TOTAL*N_TOTAL*DATA_WIDTH)-1:0] weight_store_flat;

    // Latched tile bases
    reg [$clog2(K_TOTAL)-1:0] tile_k_base_reg;
    reg [$clog2(N_TOTAL)-1:0] tile_n_base_reg;

    // Config validity
    wire tile_k_base_ok;
    wire tile_n_base_ok;
    wire start_cfg_ok;

    assign tile_k_base_ok = ((tile_k_base + TILE_K) <= K_TOTAL) &&
                            (tile_k_base[1:0] == 2'b00);

    assign tile_n_base_ok = ((tile_n_base + TILE_N) <= N_TOTAL) &&
                            (tile_n_base[1:0] == 2'b00);

    assign start_cfg_ok   = tile_k_base_ok && tile_n_base_ok;

    // ------------------------------------------------------------------------
    // Per-column injected data before skew
    // ------------------------------------------------------------------------
    reg  signed [DATA_WIDTH-1:0] inject_data [0:TILE_N-1];
    reg                          inject_valid[0:TILE_N-1];

    // Column skew pipelines:
    // column c uses stage[c] as its output
    reg signed [DATA_WIDTH-1:0] wt_pipe_data [0:TILE_N-1][0:TILE_N-1];
    reg                         wt_pipe_valid[0:TILE_N-1][0:TILE_N-1];

    integer c, s;
    integer flat_index;

    // ------------------------------------------------------------------------
    // Combinational read of current tile element to inject into stage0
    // ------------------------------------------------------------------------
    always @(*) begin
        for (c = 0; c < TILE_N; c = c + 1) begin
            inject_data[c]  = {DATA_WIDTH{1'b0}};
            inject_valid[c] = 1'b0;

            if (stream_active) begin
                flat_index = ((tile_k_base_reg + stream_step) * N_TOTAL) +
                             (tile_n_base_reg + c);

                inject_data[c]  = weight_store_flat[(flat_index*DATA_WIDTH) +: DATA_WIDTH];
                inject_valid[c] = 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Drive top-edge outputs from skew pipeline taps
    // column c takes stage c
    // ------------------------------------------------------------------------
    always @(*) begin
        col_weight_in       = {(TILE_N*DATA_WIDTH){1'b0}};
        col_weight_valid_in = {TILE_N{1'b0}};

        for (c = 0; c < TILE_N; c = c + 1) begin
            col_weight_in[(c*DATA_WIDTH) +: DATA_WIDTH] = wt_pipe_data[c][c];
            col_weight_valid_in[c]                      = wt_pipe_valid[c][c];
        end
    end

    // ------------------------------------------------------------------------
    // Sequential control + skew pipeline shift
    // Pipeline shifts every cycle so skewed data can flush during controller
    // drain cycles after the last injected stream step.
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            weight_store_flat <= {(K_TOTAL*N_TOTAL*DATA_WIDTH){1'b0}};
            tile_k_base_reg   <= {($clog2(K_TOTAL)){1'b0}};
            tile_n_base_reg   <= {($clog2(N_TOTAL)){1'b0}};
            stream_active     <= 1'b0;
            stream_step       <= {($clog2(TILE_K)){1'b0}};
            tile_done         <= 1'b0;
            cfg_error         <= 1'b0;

            for (c = 0; c < TILE_N; c = c + 1) begin
                for (s = 0; s < TILE_N; s = s + 1) begin
                    wt_pipe_data[c][s]  <= {DATA_WIDTH{1'b0}};
                    wt_pipe_valid[c][s] <= 1'b0;
                end
            end
        end
        else begin
            tile_done <= 1'b0;
            cfg_error <= 1'b0;

            if (load_en) begin
                weight_store_flat <= weight_flat;
            end

            // Shift skew pipelines every cycle
            for (c = 0; c < TILE_N; c = c + 1) begin
                for (s = TILE_N-1; s > 0; s = s - 1) begin
                    wt_pipe_data[c][s]  <= wt_pipe_data[c][s-1];
                    wt_pipe_valid[c][s] <= wt_pipe_valid[c][s-1];
                end

                wt_pipe_data[c][0]  <= inject_data[c];
                wt_pipe_valid[c][0] <= inject_valid[c];
            end

            // Start has priority over advance
            if (start_tile) begin
                if (start_cfg_ok) begin
                    tile_k_base_reg <= tile_k_base;
                    tile_n_base_reg <= tile_n_base;
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