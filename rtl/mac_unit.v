`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : mac_unit
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Registered signed multiply-accumulate unit used as a basic compute element
in the NNIA datapath.

This module multiplies one signed input activation and one signed weight,
sign-extends the product to accumulator width, and adds it to the incoming
accumulator value on the rising clock edge.

Key Behavior
------------
- Signed multiply of input_data and weight_data
- Sign extension of the product to ACC_WIDTH
- Registered accumulation into acc_out
- Synchronous reset clears acc_out to zero
- Accumulation occurs only when en is asserted

Notes
-----
- This module performs raw accumulation only
- No requantization is applied here
- No bias addition is applied here
- No activation function is applied here
- Post-processing is handled in later NNIA stages
------------------------------------------------------------------------------
*/

module mac_unit #(
    parameter integer DATA_WIDTH = 16,
    parameter integer ACC_WIDTH  = 40
)(
    input  wire                            clk,
    input  wire                            rst,
    input  wire                            en,

    input  wire signed [DATA_WIDTH-1:0]    input_data,
    input  wire signed [DATA_WIDTH-1:0]    weight_data,
    input  wire signed [ACC_WIDTH-1:0]     acc_in,

    output reg  signed [ACC_WIDTH-1:0]     acc_out
);

    // ------------------------------------------------------------------------
    // Safety check: accumulator must be wide enough to hold full product
    // ------------------------------------------------------------------------
    initial begin
        if (ACC_WIDTH < (2 * DATA_WIDTH)) begin
            $error("mac_unit: ACC_WIDTH must be >= 2*DATA_WIDTH");
            $finish;
        end
    end

    // ------------------------------------------------------------------------
    // Signed multiply
    // Q8.8 x Q8.8 -> signed 32-bit product (for DATA_WIDTH=16)
    // ------------------------------------------------------------------------
    wire signed [(2*DATA_WIDTH)-1:0] product;

    // Sign-extend product to accumulator width
    wire signed [ACC_WIDTH-1:0] product_ext;

    assign product = input_data * weight_data;
    assign product_ext = {{(ACC_WIDTH-(2*DATA_WIDTH)){product[(2*DATA_WIDTH)-1]}}, product};

    // ------------------------------------------------------------------------
    // Registered accumulate
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            acc_out <= {ACC_WIDTH{1'b0}};
        end
        else if (en) begin
            acc_out <= acc_in + product_ext;
        end
        else begin
            acc_out <= acc_out;
        end
    end

endmodule