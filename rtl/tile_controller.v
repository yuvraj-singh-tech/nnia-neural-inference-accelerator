`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : tile_controller
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Central control FSM for NNIA tiled execution.

Coordinates data streaming, accumulation, and output stages by sequencing
tile-level operations across K-dimension and output tiles. Ensures correct
timing between buffers, PE array, and storage stages.

Key Points
----------
- Controls full inference flow: stream → accumulate → save → advance
- Handles K-tile accumulation and output-tile progression
- Includes drain phase to ensure final MAC results are captured correctly
- Synchronizes input_buffer, weight_buffer, psum_buffer, and output_buffer
- Detects and flags invalid command or configuration states

Role in Pipeline
----------------
- Works alongside tile_addr_gen for execution sequencing
- Drives all major control signals across NNIA datapath
- Defines timing and ordering of tiled computation
------------------------------------------------------------------------------
*/

module tile_controller #(
    parameter integer TILE_K       = 4,
    parameter integer ROWS         = 4,
    parameter integer COLS         = 4,

    /* ------------------------------------------------------------------------
     Important:
     +1 extra cycle added beyond mesh travel to allow the final registered
     PE/MAC accumulation result to become fully visible before psum_save.
    */
    
    parameter integer DRAIN_CYCLES = ((ROWS - 1) + (COLS - 1) + 1)
)(
    input  wire clk,
    input  wire rst,

    input  wire start,
    output reg  busy,
    output reg  done,

    output reg  addr_start_run,
    output reg  addr_advance_k,
    output reg  addr_advance_out,

    input  wire first_k_tile,
    input  wire last_k_tile,
    input  wire first_output_tile,
    input  wire last_output_tile,
    input  wire addr_active,
    input  wire addr_done,
    input  wire cmd_error,

    output reg  inbuf_start,
    output reg  inbuf_advance,
    input  wire inbuf_tile_done,
    input  wire inbuf_stream_active,
    input  wire inbuf_cfg_error,

    output reg  wbuf_start,
    output reg  wbuf_advance,
    input  wire wbuf_tile_done,
    input  wire wbuf_stream_active,
    input  wire wbuf_cfg_error,

    output reg  psum_clear,
    output reg  psum_save,

    output reg  pe_clear_psum,
    output reg  pe_load_psum,

    output reg  outbuf_clear,
    output reg  outbuf_save,

    output reg  ctrl_error
);

    initial begin
        if (TILE_K <= 0) begin
            $error("tile_controller: TILE_K must be > 0.");
            $finish;
        end

        if (ROWS <= 0 || COLS <= 0) begin
            $error("tile_controller: ROWS and COLS must be > 0.");
            $finish;
        end

        if (DRAIN_CYCLES < 0) begin
            $error("tile_controller: DRAIN_CYCLES must be >= 0.");
            $finish;
        end
    end

    localparam [3:0]
        S_IDLE         = 4'd0,
        S_START_RUN    = 4'd1,
        S_CLEAR_TILE   = 4'd2,
        S_START_STREAM = 4'd3,
        S_STREAM       = 4'd4,
        S_WAIT_DONE    = 4'd5,
        S_DRAIN        = 4'd6,
        S_SAVE_PSUM    = 4'd7,
        S_ADVANCE_K    = 4'd8,
        S_SAVE_OUTPUT  = 4'd9,
        S_ADVANCE_OUT  = 4'd10,
        S_DONE         = 4'd11,
        S_ERROR        = 4'd12;

    reg [3:0] state, next_state;

    localparam integer STREAM_CNT_W =
        (TILE_K <= 1) ? 1 : $clog2(TILE_K + 1);

    localparam integer DRAIN_CNT_W =
        (DRAIN_CYCLES <= 1) ? 1 : $clog2(DRAIN_CYCLES + 1);

    reg [STREAM_CNT_W-1:0] stream_count;
    reg [DRAIN_CNT_W-1:0]  drain_count;

    reg start_d;
    wire start_rise;

    // Sticky done capture for stream buffers
    reg inbuf_tile_done_seen;
    reg wbuf_tile_done_seen;

    always @(posedge clk) begin
        if (rst)
            start_d <= 1'b0;
        else
            start_d <= start;
    end

    assign start_rise = start & ~start_d;

    always @(posedge clk) begin
        if (rst)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    always @(posedge clk) begin
        if (rst) begin
            stream_count <= {STREAM_CNT_W{1'b0}};
        end
        else begin
            case (state)
                S_START_STREAM:
                    stream_count <= {STREAM_CNT_W{1'b0}};

                S_STREAM:
                    if (stream_count < TILE_K)
                        stream_count <= stream_count + 1'b1;

                default:
                    stream_count <= stream_count;
            endcase
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            drain_count <= {DRAIN_CNT_W{1'b0}};
        end
        else begin
            case (state)
                S_WAIT_DONE:
                    drain_count <= {DRAIN_CNT_W{1'b0}};

                S_DRAIN:
                    if (DRAIN_CYCLES > 0) begin
                        if (drain_count < (DRAIN_CYCLES - 1))
                            drain_count <= drain_count + 1'b1;
                    end
                    else begin
                        drain_count <= {DRAIN_CNT_W{1'b0}};
                    end

                default:
                    drain_count <= drain_count;
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // Sticky capture of tile_done pulses so they cannot be missed
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            inbuf_tile_done_seen <= 1'b0;
            wbuf_tile_done_seen  <= 1'b0;
        end
        else begin
            case (state)
                S_START_STREAM: begin
                    inbuf_tile_done_seen <= 1'b0;
                    wbuf_tile_done_seen  <= 1'b0;
                end

                default: begin
                    if (inbuf_tile_done)
                        inbuf_tile_done_seen <= 1'b1;

                    if (wbuf_tile_done)
                        wbuf_tile_done_seen <= 1'b1;
                end
            endcase
        end
    end

    always @(*) begin
        next_state = state;

        case (state)

            S_IDLE: begin
                if (start_rise)
                    next_state = S_START_RUN;
            end

            S_START_RUN: begin
                next_state = S_CLEAR_TILE;
            end

            S_CLEAR_TILE: begin
                next_state = S_START_STREAM;
            end

            S_START_STREAM: begin
                next_state = S_STREAM;
            end

            S_STREAM: begin
                if (cmd_error || inbuf_cfg_error || wbuf_cfg_error)
                    next_state = S_ERROR;
                else if (stream_count == TILE_K)
                    next_state = S_WAIT_DONE;
            end

            S_WAIT_DONE: begin
                if (cmd_error || inbuf_cfg_error || wbuf_cfg_error)
                    next_state = S_ERROR;
                else if (inbuf_tile_done_seen && wbuf_tile_done_seen) begin
                    if (DRAIN_CYCLES == 0)
                        next_state = S_SAVE_PSUM;
                    else
                        next_state = S_DRAIN;
                end
            end

            S_DRAIN: begin
                if (drain_count == (DRAIN_CYCLES - 1))
                    next_state = S_SAVE_PSUM;
            end

            S_SAVE_PSUM: begin
                if (last_k_tile)
                    next_state = S_SAVE_OUTPUT;
                else
                    next_state = S_ADVANCE_K;
            end

            S_ADVANCE_K: begin
                next_state = S_START_STREAM;
            end

            S_SAVE_OUTPUT: begin
                next_state = S_ADVANCE_OUT;
            end

            S_ADVANCE_OUT: begin
                if (last_output_tile)
                    next_state = S_DONE;
                else
                    next_state = S_CLEAR_TILE;
            end

            S_DONE: begin
                if (!start)
                    next_state = S_IDLE;
            end

            S_ERROR: begin
                next_state = S_ERROR;
            end

            default: begin
                next_state = S_ERROR;
            end
        endcase
    end

    always @(*) begin
        addr_start_run   = 1'b0;
        addr_advance_k   = 1'b0;
        addr_advance_out = 1'b0;

        inbuf_start      = 1'b0;
        inbuf_advance    = 1'b0;

        wbuf_start       = 1'b0;
        wbuf_advance     = 1'b0;

        psum_clear       = 1'b0;
        psum_save        = 1'b0;

        pe_clear_psum    = 1'b0;
        pe_load_psum     = 1'b0;

        outbuf_clear     = 1'b0;
        outbuf_save      = 1'b0;

        busy             = 1'b1;
        done             = 1'b0;
        ctrl_error       = 1'b0;

        case (state)

            S_IDLE: begin
                busy = 1'b0;
            end

            S_START_RUN: begin
                addr_start_run = 1'b1;
                outbuf_clear   = 1'b1;
            end

            S_CLEAR_TILE: begin
                psum_clear    = 1'b1;
                pe_clear_psum = 1'b1;
            end

            S_START_STREAM: begin
                inbuf_start = 1'b1;
                wbuf_start  = 1'b1;

                if (!first_k_tile)
                    pe_load_psum = 1'b1;
            end

            S_STREAM: begin
                if (stream_count < TILE_K) begin
                    inbuf_advance = 1'b1;
                    wbuf_advance  = 1'b1;
                end
            end

            S_WAIT_DONE: begin
                // hold steady
            end

            S_DRAIN: begin
                // hold steady
            end

            S_SAVE_PSUM: begin
                psum_save = 1'b1;
            end

            S_ADVANCE_K: begin
                addr_advance_k = 1'b1;
            end

            S_SAVE_OUTPUT: begin
                outbuf_save = 1'b1;
            end

            S_ADVANCE_OUT: begin
                addr_advance_out = 1'b1;
            end

            S_DONE: begin
                busy = 1'b0;
                done = 1'b1;
            end

            S_ERROR: begin
                busy       = 1'b0;
                ctrl_error = 1'b1;
            end

            default: begin
                busy       = 1'b0;
                ctrl_error = 1'b1;
            end
        endcase
    end

endmodule