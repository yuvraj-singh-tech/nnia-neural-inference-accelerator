`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : nnia_perf_counters
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Performance monitoring unit for NNIA execution.

Captures cycle-level activity and key execution events to provide visibility
into runtime behavior, inference latency, and tiled processing progress.
Operates as a non-intrusive observer with strict separation from datapath
and control logic.

Key Insights
------------
- Distinguishes total system runtime from active inference duration
- Defines execution window using start_pulse → done_pulse
- Tracks tiled computation progress:
    • tile_writes  → output tile completion
    • psum_saves   → accumulation stage transitions
- Enables analysis of throughput, latency, and pipeline utilization

Design Characteristics
----------------------
- Fully observational; does not influence NNIA functionality
- Event-driven counting using single-cycle pulses
- Run-scoped metrics enable precise performance evaluation
- Parameterized counter width for scalable integration

Usage Context
-------------
- Instantiated at top-level (top_nnai)
- Used for simulation analysis, debugging, and reporting
- Helps validate tiled execution behavior and system efficiency
------------------------------------------------------------------------------
*/


module nnia_perf_counters #(
    parameter integer COUNT_WIDTH = 32
)(
    input  wire                    clk,
    input  wire                    rst,

    input  wire                    start_pulse,
    input  wire                    done_pulse,
    input  wire                    tile_write_pulse,
    input  wire                    psum_save_pulse,

    output reg  [COUNT_WIDTH-1:0]  total_cycles,
    output reg  [COUNT_WIDTH-1:0]  run_cycles,
    output reg  [COUNT_WIDTH-1:0]  run_count,
    output reg  [COUNT_WIDTH-1:0]  tile_writes,
    output reg  [COUNT_WIDTH-1:0]  psum_saves
);

    reg run_active;

    always @(posedge clk) begin
        if (rst) begin
            run_active   <= 1'b0;
            total_cycles <= {COUNT_WIDTH{1'b0}};
            run_cycles   <= {COUNT_WIDTH{1'b0}};
            run_count    <= {COUNT_WIDTH{1'b0}};
            tile_writes  <= {COUNT_WIDTH{1'b0}};
            psum_saves   <= {COUNT_WIDTH{1'b0}};
        end
        else begin
            total_cycles <= total_cycles + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};

            if (start_pulse) begin
                run_active <= 1'b1;
                run_cycles <= {COUNT_WIDTH{1'b0}};
            end
            else if (done_pulse) begin
                run_active <= 1'b0;
                run_count  <= run_count + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};
            end

            if (run_active)
                run_cycles <= run_cycles + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};

            if (tile_write_pulse)
                tile_writes <= tile_writes + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};

            if (psum_save_pulse)
                psum_saves <= psum_saves + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};
        end
    end

endmodule
