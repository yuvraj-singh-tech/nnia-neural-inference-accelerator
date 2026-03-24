`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : psum_buffer
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Tile-level partial sum storage for NNIA accumulation flow.

Stores and restores a full 4x4 tile of intermediate accumulation results,
enabling multi-step tiled computation across K-dimension iterations.

Key Points
----------
- Holds partial sums between successive accumulation phases
- Provides psum_seed back to PE array for continued computation
- buf_valid indicates presence of stored accumulation state
- clear and save controls define psum lifecycle

Role in Pipeline
----------------
- Positioned between pe_array_4x4 and postprocess_array
- Enables multi-tile accumulation before final post-processing
------------------------------------------------------------------------------
*/

module psum_buffer #(
    parameter integer ROWS      = 4,
    parameter integer COLS      = 4,
    parameter integer ACC_WIDTH = 40
)(
    input  wire                                     clk,
    input  wire                                     rst,

    // ------------------------------------------------------------------------
    // Buffer control
    // ------------------------------------------------------------------------
    input  wire                                     clear_buf,
    input  wire                                     save_en,

    // ------------------------------------------------------------------------
    // Incoming updated psums from pe_array_4x4
    // One psum per PE, flattened row-major
    // ------------------------------------------------------------------------
    input  wire signed [(ROWS*COLS*ACC_WIDTH)-1:0]  psum_in_flat,

    // ------------------------------------------------------------------------
    // Stored psum state exposed back to pe_array_4x4 as seed input
    // ------------------------------------------------------------------------
    output wire signed [(ROWS*COLS*ACC_WIDTH)-1:0]  psum_seed_flat,

    // ------------------------------------------------------------------------
    // Stored psum state exposed for observation / later post-processing
    // ------------------------------------------------------------------------
    output wire signed [(ROWS*COLS*ACC_WIDTH)-1:0]  psum_out_flat,

    // ------------------------------------------------------------------------
    // Indicates whether buffer currently contains a valid saved tile
    // ------------------------------------------------------------------------
    output reg                                      buf_valid
);

    // ------------------------------------------------------------------------
    // Safety checks
    // ------------------------------------------------------------------------
    initial begin
        if (ROWS != 4 || COLS != 4) begin
            $error("psum_buffer: This module is intended for a 4x4 psum tile.");
            $finish;
        end

        if (ACC_WIDTH <= 0) begin
            $error("psum_buffer: ACC_WIDTH must be > 0");
            $finish;
        end
    end

    // ------------------------------------------------------------------------
    // Internal storage: one signed accumulator register per tile position
    // Flattened row-major to match pe_array_4x4 exactly
    // ------------------------------------------------------------------------
    reg signed [(ROWS*COLS*ACC_WIDTH)-1:0] psum_store_flat;

    // ------------------------------------------------------------------------
    // Synchronous state storage
    // clear_buf has priority over save_en
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            psum_store_flat <= {(ROWS*COLS*ACC_WIDTH){1'b0}};
            buf_valid       <= 1'b0;
        end
        else if (clear_buf) begin
            psum_store_flat <= {(ROWS*COLS*ACC_WIDTH){1'b0}};
            buf_valid       <= 1'b0;
        end
        else if (save_en) begin
            psum_store_flat <= psum_in_flat;
            buf_valid       <= 1'b1;
        end
    end

    // ------------------------------------------------------------------------
    // Outputs
    // Stored state is directly used as:
    // - psum seed for the PE array
    // - observable raw psum tile output
    // ------------------------------------------------------------------------
    assign psum_seed_flat = psum_store_flat;
    assign psum_out_flat  = psum_store_flat;

endmodule