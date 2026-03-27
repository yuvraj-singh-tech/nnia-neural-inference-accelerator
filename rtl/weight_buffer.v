/*
------------------------------------------------------------------------------
Module Name : weight_buffer
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Weight storage and streaming stage for NNIA tiled execution.

This module stores the full weight matrix, selects the required tile window,
and streams one weight vector per cycle into the PE array. A column-wise skew
pipeline is applied to ensure correct temporal alignment with activation flow.

Key Points
----------
- Streams weights per K-step into PE array
- Applies column-wise skew for correct alignment
- Maintains tile configuration and streaming control
- tile_done indicates completion of one tile stream
- cfg_error flags invalid tile configuration

Role in Pipeline
----------------
- Works with input_buffer for synchronized data injection
- Feeds pe_array_4x4 with aligned weight data
- Supports tiled accumulation across K dimension
------------------------------------------------------------------------------
*/

`timescale 1ns / 1ps

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

    // ------------------------------------------------------------------------
    // Storage
    // ------------------------------------------------------------------------
    reg signed [(K_TOTAL*N_TOTAL*DATA_WIDTH)-1:0] weight_store_flat;

    // Active tile
    reg [$clog2(K_TOTAL)-1:0] tile_k_base_reg;
    reg [$clog2(N_TOTAL)-1:0] tile_n_base_reg;

    //  Pending tile 
    reg [$clog2(K_TOTAL)-1:0] pending_tile_k_base;
    reg [$clog2(N_TOTAL)-1:0] pending_tile_n_base;
    reg                       pending_valid;

    // ------------------------------------------------------------------------
    // Config check
    // ------------------------------------------------------------------------
    wire start_cfg_ok;

    assign start_cfg_ok =
        ((tile_k_base + TILE_K) <= K_TOTAL) &&
        ((tile_n_base + TILE_N) <= N_TOTAL) &&
        (tile_k_base[1:0] == 2'b00) &&
        (tile_n_base[1:0] == 2'b00);

    // ------------------------------------------------------------------------
    // Injection
    // ------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] inject_data  [0:TILE_N-1];
    reg                         inject_valid[0:TILE_N-1];

    reg signed [DATA_WIDTH-1:0] wt_pipe_data  [0:TILE_N-1][0:TILE_N-1];
    reg                         wt_pipe_valid [0:TILE_N-1][0:TILE_N-1];

    integer c, s;
    integer flat_index;

    always @(*) begin
        for (c = 0; c < TILE_N; c = c + 1) begin
            inject_data[c]  = 0;
            inject_valid[c] = 0;

            if (stream_active) begin
                flat_index = ((tile_k_base_reg + stream_step) * N_TOTAL) +
                             (tile_n_base_reg + c);

                inject_data[c]  = weight_store_flat[(flat_index*DATA_WIDTH)+:DATA_WIDTH];
                inject_valid[c] = 1'b1;
            end
        end
    end

    always @(*) begin
        col_weight_in       = 0;
        col_weight_valid_in = 0;

        for (c = 0; c < TILE_N; c = c + 1) begin
            col_weight_in[(c*DATA_WIDTH)+:DATA_WIDTH] = wt_pipe_data[c][c];
            col_weight_valid_in[c]                    = wt_pipe_valid[c][c];
        end
    end

    // ------------------------------------------------------------------------
    // Main logic
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            weight_store_flat <= 0;
            tile_k_base_reg   <= 0;
            tile_n_base_reg   <= 0;

            pending_tile_k_base <= 0;
            pending_tile_n_base <= 0;
            pending_valid       <= 0;

            stream_active <= 0;
            stream_step   <= 0;
            tile_done     <= 0;
            cfg_error     <= 0;

            for (c = 0; c < TILE_N; c = c + 1)
                for (s = 0; s < TILE_N; s = s + 1) begin
                    wt_pipe_data[c][s]  <= 0;
                    wt_pipe_valid[c][s] <= 0;
                end
        end
        else begin
            tile_done <= 0;
            cfg_error <= 0;

            if (load_en)
                weight_store_flat <= weight_flat;

            // shift pipeline
            for (c = 0; c < TILE_N; c = c + 1) begin
                for (s = TILE_N-1; s > 0; s = s - 1) begin
                    wt_pipe_data[c][s]  <= wt_pipe_data[c][s-1];
                    wt_pipe_valid[c][s] <= wt_pipe_valid[c][s-1];
                end

                wt_pipe_data[c][0]  <= inject_data[c];
                wt_pipe_valid[c][0] <= inject_valid[c];
            end

            // 🔥 START / QUEUE LOGIC
            if (start_tile) begin
                if (start_cfg_ok) begin
                    if (!stream_active) begin
                        // start immediately
                        tile_k_base_reg <= tile_k_base;
                        tile_n_base_reg <= tile_n_base;
                        stream_active   <= 1;
                        stream_step     <= 0;
                    end
                    else begin
                        // queue next tile
                        pending_tile_k_base <= tile_k_base;
                        pending_tile_n_base <= tile_n_base;
                        pending_valid       <= 1;
                    end
                end
                else begin
                    cfg_error <= 1;
                end
            end

            // advance
            else if (stream_active && advance) begin
                if (stream_step == TILE_K-1) begin
                    tile_done <= 1;

                    if (pending_valid) begin
                        // 🔥 instant switch (NO GAP)
                        tile_k_base_reg <= pending_tile_k_base;
                        tile_n_base_reg <= pending_tile_n_base;
                        stream_step     <= 0;
                        pending_valid   <= 0;
                        stream_active   <= 1;
                    end
                    else begin
                        stream_active <= 0;
                        stream_step   <= 0;
                    end
                end
                else begin
                    stream_step <= stream_step + 1;
                end
            end
        end
    end

endmodule
