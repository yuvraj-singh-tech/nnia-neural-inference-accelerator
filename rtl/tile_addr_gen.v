`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : tile_addr_gen
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Tile-level coordinate generator for NNIA execution control.

Produces base indices for activation, weight, and output tiles, enabling
deterministic traversal across K-dimension accumulation and output tiles
without relying on traditional memory addressing.

Key Points
----------
- Drives tiled execution sequencing (K-loop and output-tile loop)
- Generates tile_m_base, tile_k_base, tile_n_base for all buffers
- Ensures correct progression: K-tiles → output tile → next tile
- Provides status flags (first/last tile) for controller decisions
- Includes command validation and error detection

Role in Pipeline
----------------
- Works with controller_fsm to manage full inference flow
- Coordinates input_buffer, weight_buffer, and accumulation stages
- Defines execution order of NNIA tiled computation
------------------------------------------------------------------------------
*/

module tile_addr_gen #(
    parameter integer M_TOTAL = 4,
    parameter integer K_TOTAL = 16,
    parameter integer N_TOTAL = 8,
    parameter integer TILE_M  = 4,
    parameter integer TILE_K  = 4,
    parameter integer TILE_N  = 4
)(
    input  wire                              clk,
    input  wire                              rst,

    // Controller commands
    input  wire                              start_run,
    input  wire                              advance_k_tile,
    input  wire                              advance_output_tile,

    // Tile-base outputs toward buffers / top-level wiring
    output wire [$clog2(M_TOTAL)-1:0]        tile_m_base,
    output wire [$clog2(K_TOTAL)-1:0]        tile_k_base,
    output wire [$clog2(N_TOTAL)-1:0]        tile_n_base,

    // Status / debug outputs
    output wire                              first_k_tile,
    output wire                              last_k_tile,
    output wire                              first_output_tile,
    output wire                              last_output_tile,

    output reg                               active,
    output reg                               all_done,
    output reg                               cmd_error
);

    // ------------------------------------------------------------------------
    // Tile-count derivation
    // ------------------------------------------------------------------------
    localparam integer M_TILES = M_TOTAL / TILE_M;
    localparam integer K_TILES = K_TOTAL / TILE_K;
    localparam integer N_TILES = N_TOTAL / TILE_N;

    localparam integer K_TILE_IDX_W = (K_TILES <= 1) ? 1 : $clog2(K_TILES);
    localparam integer N_TILE_IDX_W = (N_TILES <= 1) ? 1 : $clog2(N_TILES);

    // Internal tile indices
    reg [K_TILE_IDX_W-1:0] k_tile_idx;
    reg [N_TILE_IDX_W-1:0] n_tile_idx;

    // Command conflict detection: only one command allowed per cycle
    wire multi_cmd_error;
    assign multi_cmd_error =
        (start_run && advance_k_tile)      ||
        (start_run && advance_output_tile) ||
        (advance_k_tile && advance_output_tile);

    // ------------------------------------------------------------------------
    // Safety checks for locked NNIA architecture
    // ------------------------------------------------------------------------
    initial begin
        if (M_TOTAL != 4) begin
            $error("tile_addr_gen: Locked NNIA version expects M_TOTAL=4.");
            $finish;
        end

        if (K_TOTAL != 16) begin
            $error("tile_addr_gen: Locked NNIA version expects K_TOTAL=16.");
            $finish;
        end

        if (N_TOTAL != 8) begin
            $error("tile_addr_gen: Locked NNIA version expects N_TOTAL=8.");
            $finish;
        end

        if (TILE_M != 4) begin
            $error("tile_addr_gen: Locked NNIA version expects TILE_M=4.");
            $finish;
        end

        if (TILE_K != 4) begin
            $error("tile_addr_gen: Locked NNIA version expects TILE_K=4.");
            $finish;
        end

        if (TILE_N != 4) begin
            $error("tile_addr_gen: Locked NNIA version expects TILE_N=4.");
            $finish;
        end

        if ((M_TOTAL % TILE_M) != 0) begin
            $error("tile_addr_gen: M_TOTAL must be divisible by TILE_M.");
            $finish;
        end

        if ((K_TOTAL % TILE_K) != 0) begin
            $error("tile_addr_gen: K_TOTAL must be divisible by TILE_K.");
            $finish;
        end

        if ((N_TOTAL % TILE_N) != 0) begin
            $error("tile_addr_gen: N_TOTAL must be divisible by TILE_N.");
            $finish;
        end

        if (M_TILES != 1) begin
            $error("tile_addr_gen: Locked NNIA version expects exactly one M tile.");
            $finish;
        end
    end

    // ------------------------------------------------------------------------
    // Current tile-base outputs
    // Locked NNIA:
    // - tile_m_base always 0
    // - TILE_K = 4 => multiply by 4 implemented as << 2
    // - TILE_N = 4 => multiply by 4 implemented as << 2
    // ------------------------------------------------------------------------
    assign tile_m_base = {($clog2(M_TOTAL)){1'b0}};
    assign tile_k_base = {k_tile_idx, 2'b00};
    assign tile_n_base = {n_tile_idx, 2'b00};

    // Status flags for controller sequencing
    assign first_k_tile      = (k_tile_idx == {K_TILE_IDX_W{1'b0}});
    assign last_k_tile       = (k_tile_idx == (K_TILES-1));
    assign first_output_tile = (n_tile_idx == {N_TILE_IDX_W{1'b0}});
    assign last_output_tile  = (n_tile_idx == (N_TILES-1));

    // ------------------------------------------------------------------------
    // Sequential control
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            k_tile_idx <= {K_TILE_IDX_W{1'b0}};
            n_tile_idx <= {N_TILE_IDX_W{1'b0}};
            active     <= 1'b0;
            all_done   <= 1'b0;
            cmd_error  <= 1'b0;
        end
        else begin
            cmd_error <= 1'b0;

            if (multi_cmd_error) begin
                cmd_error <= 1'b1;
            end
            else if (start_run) begin
                if (active) begin
                    cmd_error <= 1'b1;
                end
                else begin
                    k_tile_idx <= {K_TILE_IDX_W{1'b0}};
                    n_tile_idx <= {N_TILE_IDX_W{1'b0}};
                    active     <= 1'b1;
                    all_done   <= 1'b0;
                end
            end
            else if (advance_k_tile) begin
                if (!active || all_done) begin
                    cmd_error <= 1'b1;
                end
                else if (last_k_tile) begin
                    cmd_error <= 1'b1;
                end
                else begin
                    k_tile_idx <= k_tile_idx + 1'b1;
                end
            end
            else if (advance_output_tile) begin
                if (!active || all_done) begin
                    cmd_error <= 1'b1;
                end
                else if (!last_k_tile) begin
                    cmd_error <= 1'b1;
                end
                else if (last_output_tile) begin
                    active   <= 1'b0;
                    all_done <= 1'b1;
                end
                else begin
                    n_tile_idx <= n_tile_idx + 1'b1;
                    k_tile_idx <= {K_TILE_IDX_W{1'b0}};
                end
            end
        end
    end

endmodule