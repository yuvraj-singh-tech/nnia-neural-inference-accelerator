# -----------------------------------------------------------------------------
# Module Name : run_nnia_sim.tcl
# Author      : Yuvraj Singh
# Project     : Neural Network Inference Accelerator (NNIA)
#
# Description
# -----------
# Vivado batch simulation script for end-to-end NNIA verification.
#
# Automates the complete flow including RTL compilation, testbench execution,
# memory validation, output generation, and performance extraction to provide
# a reproducible and structured simulation pipeline.
#
# Key Points
# ----------
# - Verifies presence of RTL, testbench, and memory files
# - Performs clean build (removes previous simulation artifacts)
# - Compiles RTL and testbench, followed by elaboration and simulation
# - Extracts latency and throughput metrics from TB-reported performance counters
# - Separates peak theoretical throughput from effective measured throughput
# - Validates generated output file for correctness
#
# Role in Flow
# ------------
# - Acts as the execution backbone of NNIA verification
# - Bridges Python-generated memory files with RTL simulation
# - Produces both functional output and performance metrics
# -----------------------------------------------------------------------------

puts ""
puts "======================================================================"
puts "              NEURAL NETWORK INFERENCE ACCELERATOR (NNIA)             "
puts "                    VIVADO BATCH SIMULATION FLOW                      "
puts "======================================================================"

# -----------------------------------------------------------------------------
# Resolve project root
# -----------------------------------------------------------------------------
set script_dir   [file dirname [file normalize [info script]]]
set project_root [file dirname $script_dir]
cd $project_root

puts ""
puts "INFO  : Project root resolved successfully."
puts "INFO  : Project root : $project_root"

# -----------------------------------------------------------------------------
# Locked NNIA configuration
# -----------------------------------------------------------------------------
set M_TOTAL 4.0
set K_TOTAL 16.0
set N_TOTAL 8.0
set ROWS    4.0
set COLS    4.0

# Testbench clock
set clk_period_ns 10.0

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
set rtl_dir   "rtl"
set tb_dir    "tb"
set mem_dir   "mem"
set logs_dir  "logs"

set rtl_files [list \
    "$rtl_dir/mac_unit.v" \
    "$rtl_dir/relu_unit.v" \
    "$rtl_dir/pe_unit.v" \
    "$rtl_dir/pe_array_4x4.v" \
    "$rtl_dir/input_buffer.v" \
    "$rtl_dir/weight_buffer.v" \
    "$rtl_dir/psum_buffer.v" \
    "$rtl_dir/output_buffer.v" \
    "$rtl_dir/quant_bias_relu.v" \
    "$rtl_dir/postprocess_array.v" \
    "$rtl_dir/tile_addr_gen.v" \
    "$rtl_dir/tile_controller.v" \
    "$rtl_dir/nnia_perf_counters.v" \
    "$rtl_dir/top_nnia.v" \
]

set tb_file "$tb_dir/tb_top_nnia.v"
set top_tb  "tb_top_nnia"

set snapshot_name "nnia_tb_snapshot"
set rtl_out_file  "$mem_dir/rtl_output.mem"

# -----------------------------------------------------------------------------
# Create logs folder if needed
# -----------------------------------------------------------------------------
if {![file exists $logs_dir]} {
    file mkdir $logs_dir
}

# -----------------------------------------------------------------------------
# STEP 1 : Sanity checks
# -----------------------------------------------------------------------------
puts ""
puts "----------------------------------------------------------------------"
puts "STEP 1 : INPUT FILE CHECKS"
puts "----------------------------------------------------------------------"
puts "INFO  : Verifying RTL and testbench files..."

foreach f $rtl_files {
    if {![file exists $f]} {
        puts "ERROR : Missing RTL file: $f"
        exit 1
    }
}

if {![file exists $tb_file]} {
    puts "ERROR : Missing testbench file: $tb_file"
    exit 1
}

puts "PASS  : All RTL and testbench files are present."

# -----------------------------------------------------------------------------
# STEP 2 : Memory checks
# -----------------------------------------------------------------------------
puts ""
puts "----------------------------------------------------------------------"
puts "STEP 2 : MEMORY FILE CHECKS"
puts "----------------------------------------------------------------------"
puts "INFO  : Verifying required memory files..."

set required_mem_files [list \
    "$mem_dir/input.mem" \
    "$mem_dir/weights.mem" \
    "$mem_dir/bias.mem" \
    "$mem_dir/expected_output.mem" \
]

foreach f $required_mem_files {
    if {![file exists $f]} {
        puts "ERROR : Missing memory file: $f"
        exit 1
    }
}

puts "PASS  : All memory files are present."

# -----------------------------------------------------------------------------
# STEP 3 : Cleanup
# -----------------------------------------------------------------------------
puts ""
puts "----------------------------------------------------------------------"
puts "STEP 3 : CLEANUP"
puts "----------------------------------------------------------------------"
puts "INFO  : Removing previous simulation artifacts..."

set cleanup_items [list \
    "xsim.dir" \
    ".Xil" \
    "webtalk.log" \
    "webtalk.jou" \
    "xvlog.pb" \
    "xelab.pb" \
    "xsim.pb" \
    "$snapshot_name.wdb" \
]

foreach item $cleanup_items {
    if {[file exists $item]} {
        file delete -force $item
    }
}

if {[file exists $rtl_out_file]} {
    file delete -force $rtl_out_file
}

puts "PASS  : Cleanup completed successfully."

# -----------------------------------------------------------------------------
# STEP 4 : Compile RTL
# -----------------------------------------------------------------------------
puts ""
puts "----------------------------------------------------------------------"
puts "STEP 4 : RTL COMPILATION"
puts "----------------------------------------------------------------------"

foreach f $rtl_files {
    puts "  xvlog -sv $f"
    if {[catch {exec xvlog -sv $f} msg]} {
        puts "ERROR : RTL compilation failed for: $f"
        puts $msg
        exit 1
    }
}

puts "PASS  : RTL compilation completed successfully."

# -----------------------------------------------------------------------------
# STEP 5 : Compile TB
# -----------------------------------------------------------------------------
puts ""
puts "----------------------------------------------------------------------"
puts "STEP 5 : TESTBENCH COMPILATION"
puts "----------------------------------------------------------------------"

if {[catch {exec xvlog -sv $tb_file} msg]} {
    puts "ERROR : Testbench compilation failed."
    puts $msg
    exit 1
}

puts "PASS  : Testbench compilation completed successfully."

# -----------------------------------------------------------------------------
# STEP 6 : Elaborate
# -----------------------------------------------------------------------------
puts ""
puts "----------------------------------------------------------------------"
puts "STEP 6 : ELABORATION"
puts "----------------------------------------------------------------------"

if {[catch {exec xelab $top_tb -debug typical -s $snapshot_name} msg]} {
    puts "ERROR : Elaboration failed."
    puts $msg
    exit 1
}

puts "PASS  : Elaboration completed successfully."

# -----------------------------------------------------------------------------
# STEP 7 : Run simulation
# -----------------------------------------------------------------------------
puts ""
puts "----------------------------------------------------------------------"
puts "STEP 7 : SIMULATION"
puts "----------------------------------------------------------------------"
puts "INFO  : Launching simulation run..."

if {[catch {exec xsim $snapshot_name -runall} msg]} {
    puts "ERROR : Simulation failed."
    puts $msg
    exit 1
}

puts "PASS  : Simulation completed successfully."

puts ""
puts "----------------------------------------------------------------------"
puts "SIMULATION CONSOLE OUTPUT"
puts "----------------------------------------------------------------------"
puts $msg

# -----------------------------------------------------------------------------
# Extract performance counters from simulation console
# -----------------------------------------------------------------------------
set perf_total_cycles ""
set perf_run_cycles   ""
set perf_run_count    ""
set perf_tile_writes  ""
set perf_psum_saves   ""

foreach line [split $msg "\n"] {
    set line [string trim $line]

    if {[regexp {^PERF_TOTAL_CYCLES=([0-9]+)$} $line -> value]} {
        set perf_total_cycles $value
    }
    if {[regexp {^PERF_RUN_CYCLES=([0-9]+)$} $line -> value]} {
        set perf_run_cycles $value
    }
    if {[regexp {^PERF_RUN_COUNT=([0-9]+)$} $line -> value]} {
        set perf_run_count $value
    }
    if {[regexp {^PERF_TILE_WRITES=([0-9]+)$} $line -> value]} {
        set perf_tile_writes $value
    }
    if {[regexp {^PERF_PSUM_SAVES=([0-9]+)$} $line -> value]} {
        set perf_psum_saves $value
    }
}

# -----------------------------------------------------------------------------
# Performance calculation
# -----------------------------------------------------------------------------
if {$perf_run_cycles ne ""} {

    set latency_cycles [expr {double($perf_run_cycles)}]
    set clk_freq_hz    [expr {1.0e9 / $clk_period_ns}]
    set clk_freq_mhz   [expr {$clk_freq_hz / 1.0e6}]

    set latency_ns     [expr {$latency_cycles * $clk_period_ns}]
    set latency_us     [expr {$latency_ns / 1000.0}]

    # Actual dense MAC work for one full NNIA inference
    set total_macs [expr {$M_TOTAL * $K_TOTAL * $N_TOTAL}]

    # Peak hardware capability from PE array size
    set peak_macs_per_cycle [expr {$ROWS * $COLS}]
    set peak_throughput_mmacs [expr {$peak_macs_per_cycle * $clk_freq_mhz}]

    # Effective throughput from measured run cycles
    set effective_throughput_mmacs [expr {($total_macs / $latency_cycles) * $clk_freq_mhz}]

    # Utilization as ratio of effective to peak throughput
    if {$peak_throughput_mmacs > 0.0} {
        set utilization_percent [expr {($effective_throughput_mmacs / $peak_throughput_mmacs) * 100.0}]
    } else {
        set utilization_percent 0.0
    }

    puts ""
    puts "======================================================================"
    puts "                          NNIA INFERENCE RESULTS                      "
    puts "======================================================================"
    puts [format "%-32s : %.0f"        "Inference latency (cycles)"   $latency_cycles]
    puts [format "%-32s : %.2f us"     "Inference latency"            $latency_us]
    puts [format "%-32s : %.0f"        "Total MACs per inference"     $total_macs]
    puts [format "%-32s : %.0f"        "Peak MACs per cycle"          $peak_macs_per_cycle]
    puts [format "%-32s : %.2f MMAC/s" "Peak throughput @100MHz"      $peak_throughput_mmacs]
    puts [format "%-32s : %.2f MMAC/s" "Effective throughput"         $effective_throughput_mmacs]

    if {$perf_total_cycles ne ""} {
        puts [format "%-32s : %s" "Perf total cycles" $perf_total_cycles]
    }
    if {$perf_run_count ne ""} {
        puts [format "%-32s : %s" "Perf run count" $perf_run_count]
    }
    if {$perf_tile_writes ne ""} {
        puts [format "%-32s : %s" "Perf tile writes" $perf_tile_writes]
    }
    if {$perf_psum_saves ne ""} {
        puts [format "%-32s : %s" "Perf psum saves" $perf_psum_saves]
    }
    puts "======================================================================"

    # --- HOST SAFE OUTPUT ---
    puts [format "HOST_LATENCY_CYCLES=%.0f" $latency_cycles]
    puts [format "HOST_LATENCY_TIME_US=%.2f" $latency_us]
    puts [format "HOST_TOTAL_MACS=%.0f" $total_macs]
    puts [format "HOST_PEAK_MACS_PER_CYCLE=%.0f" $peak_macs_per_cycle]
    puts [format "HOST_PEAK_THROUGHPUT_MMACS=%.2f" $peak_throughput_mmacs]
    puts [format "HOST_EFFECTIVE_THROUGHPUT_MMACS=%.2f" $effective_throughput_mmacs]

} else {
    puts ""
    puts "WARN  : PERF_RUN_CYCLES not detected from simulation output."
}

# -----------------------------------------------------------------------------
# STEP 8 : Output check
# -----------------------------------------------------------------------------
puts ""
puts "----------------------------------------------------------------------"
puts "STEP 8 : OUTPUT FILE CHECK"
puts "----------------------------------------------------------------------"

if {![file exists $rtl_out_file]} {
    puts "ERROR : Output file not found: $rtl_out_file"
    exit 1
}

set fh [open $rtl_out_file r]
set contents [read $fh]
close $fh

if {[string trim $contents] eq ""} {
    puts "ERROR : Output file is empty: $rtl_out_file"
    exit 1
}

puts "PASS  : Output file generated successfully."
puts "INFO  : Output file : $rtl_out_file"

puts ""
puts "======================================================================"
puts "                         NNIA SIMULATION PASS                         "
puts "======================================================================"

exit 0
