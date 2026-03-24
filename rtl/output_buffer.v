`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : output_buffer
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Registered output tile storage for NNIA tiled execution.

Captures the fully processed 4x4 output tile from the post-processing stage
and holds it stable for downstream consumption (writeback, verification, or
external observation). Ensures clean timing isolation between compute and
output stages.

Key Insights
------------
- Acts as the final stabilization point for each computed output tile
- Decouples combinational post-processing from external interfaces
- Provides explicit tile commit event via tile_saved_pulse
- buf_valid indicates availability of a complete, valid output tile

Design Characteristics
----------------------
- Synchronous register-based storage of full tile 
- Clear-before-use control ensures deterministic tile lifecycle
- Single-cycle save operation aligned with controller scheduling
- Fixed 4x4 tile assumption ensures strict architectural alignment

Usage Context
-------------
- Placed after postprocess_array in NNIA pipeline
- Interfaces with controller for tile-level writeback sequencing
- Used by testbench and host flow for capturing RTL outputs
------------------------------------------------------------------------------
*/

module output_buffer #(
    parameter integer ROWS       = 4,
    parameter integer COLS       = 4,
    parameter integer DATA_WIDTH = 16
)(
    input  wire                                       clk,
    input  wire                                       rst,

    input  wire                                       clear_buf,
    input  wire                                       save_en,

    input  wire signed [(ROWS*COLS*DATA_WIDTH)-1:0]   out_tile_in_flat,

    output wire signed [(ROWS*COLS*DATA_WIDTH)-1:0]   out_tile_flat,
    output reg                                        buf_valid,
    output reg                                        tile_saved_pulse
);

    initial begin
        if (ROWS != 4 || COLS != 4) begin
            $error("output_buffer: This module is intended for a 4x4 output tile.");
            $finish;
        end

        if (DATA_WIDTH <= 0) begin
            $error("output_buffer: DATA_WIDTH must be > 0.");
            $finish;
        end
    end

    reg signed [(ROWS*COLS*DATA_WIDTH)-1:0] out_store_flat;

    always @(posedge clk) begin
        if (rst) begin
            out_store_flat     <= {(ROWS*COLS*DATA_WIDTH){1'b0}};
            buf_valid          <= 1'b0;
            tile_saved_pulse   <= 1'b0;
        end
        else begin
            tile_saved_pulse <= 1'b0;

            if (clear_buf) begin
                out_store_flat   <= {(ROWS*COLS*DATA_WIDTH){1'b0}};
                buf_valid        <= 1'b0;
            end
            else if (save_en) begin
                out_store_flat   <= out_tile_in_flat;
                buf_valid        <= 1'b1;
                tile_saved_pulse <= 1'b1;
            end
        end
    end

    assign out_tile_flat = out_store_flat;

endmodule