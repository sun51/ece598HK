#set ylabel "ms/step" 
set ylabel "speedup" font "Times-Roman,22,bold"
#set xlabel "number of cores" 
set xlabel "number of cpu cores" font "Times-Roman,20,bold"
#set title "1M-atom simulation (PME every 4 steps) time step on Jaguar and Titan" 
#set title "100M-atom on Jaguar and Blue Wasters" font "Times-Roman,15" 
set logscale x 2
set logscale y 2
#set autoscale
set key top right 
set xrange [1:16]
set yrange [1:16]
set ytics  0.2
set size 0.7, 0.7
set style line   1 lt 1 lw 3 
set style line   2 lt 3 lw 3 
set style line   3 lt 2 lw 3 
set grid noxtics ytics
#set xtics ("4K" 4096, "16K" 16384, "64K" 65536, "128K" 131072,  "298992" 298992)
#set terminal png size 800 500
#set output "jaguar_ess.png"
set terminal postscript eps enhanced color "NimbusSanL-Regu" 
set output "gpu-singlenode.eps"

plot "gpu-1node.dat"  using 2:(113.6/$3)  with linespoints ls 2 title "sing node performance" 

