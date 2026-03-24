`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : postprocess_array
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Tile-level post-processing stage for NNIA.

Applies final transformations on a full 4x4 raw accumulator tile in parallel
using multiple quant_bias_relu lanes, converting raw compute outputs into
final inference results.

Key Points
----------
- Processes entire tile in parallel (one lane per output)
- Uses column-wise bias mapping across all rows
- Pure combinational stage (no state or timing control)
- Maintains strict alignment with software inference flow

Role in Pipeline
----------------
- Placed after psum_buffer
- Produces final DATA_WIDTH outputs before output_buffer
------------------------------------------------------------------------------
*/

module postprocess_array #(
    parameter integer ROWS       = 4,
    parameter integer COLS       = 4,
    parameter integer DATA_WIDTH = 16,
    parameter integer FRAC_BITS  = 8,
    parameter integer ACC_WIDTH  = 40
)(
    // ------------------------------------------------------------------------
    // One full raw accumulator tile, flattened row-major
    // ------------------------------------------------------------------------
    input  wire signed [(ROWS*COLS*ACC_WIDTH)-1:0]  raw_acc_tile_flat,

    // ------------------------------------------------------------------------
    // One bias per output column
    // col_bias_in[col]
    // ------------------------------------------------------------------------
    input  wire signed [(COLS*DATA_WIDTH)-1:0]      col_bias_in,

    // ------------------------------------------------------------------------
    // One final output per tile position, flattened row-major
    // ------------------------------------------------------------------------
    output wire signed [(ROWS*COLS*DATA_WIDTH)-1:0] out_tile_flat
);

    // ------------------------------------------------------------------------
    // Safety checks
    // ------------------------------------------------------------------------
    initial begin
        if (ROWS != 4 || COLS != 4) begin
            $error("postprocess_array: This module is intended for a 4x4 tile.");
            $finish;
        end

        if (ACC_WIDTH < DATA_WIDTH) begin
            $error("postprocess_array: ACC_WIDTH must be >= DATA_WIDTH.");
            $finish;
        end

        if (DATA_WIDTH <= 0 || ACC_WIDTH <= 0) begin
            $error("postprocess_array: DATA_WIDTH and ACC_WIDTH must be > 0.");
            $finish;
        end

        if (FRAC_BITS < 0) begin
            $error("postprocess_array: FRAC_BITS must be >= 0.");
            $finish;
        end
    end

    // ------------------------------------------------------------------------
    // 16 parallel scalar postprocess lanes
    // Each lane uses:
    //   raw_acc_tile_flat[row,col]
    //   bias of that output column
    // ------------------------------------------------------------------------
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : GEN_ROWS
            for (c = 0; c < COLS; c = c + 1) begin : GEN_COLS
                localparam integer TILE_INDEX = (r * COLS) + c;

                wire signed [ACC_WIDTH-1:0]  lane_raw_acc;
                wire signed [DATA_WIDTH-1:0] lane_bias;
                wire signed [DATA_WIDTH-1:0] lane_out;

                assign lane_raw_acc =
                    raw_acc_tile_flat[(TILE_INDEX*ACC_WIDTH) +: ACC_WIDTH];

                assign lane_bias =
                    col_bias_in[(c*DATA_WIDTH) +: DATA_WIDTH];

                quant_bias_relu #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .FRAC_BITS (FRAC_BITS),
                    .ACC_WIDTH (ACC_WIDTH)
                ) u_quant_bias_relu (
                    .raw_acc_in (lane_raw_acc),
                    .bias_data  (lane_bias),
                    .out_data   (lane_out)
                );

                assign out_tile_flat[(TILE_INDEX*DATA_WIDTH) +: DATA_WIDTH] =
                    lane_out;
            end
        end
    endgenerate

endmodule