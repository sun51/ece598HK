#set ylabel "ms/step" 
set ylabel "ms/step" font "Times-Roman,22,bold"
#set xlabel "number of cores" 
set xlabel "number of nodes" font "Times-Roman,20,bold"
#set title "100M-atom on Jaguar and Blue Wasters" font "Times-Roman,15" 
set logscale x 2
set logscale y 2
#set autoscale
set key top right 
set xrange [1:32]
set yrange [1:100]
set ytics  0.2
set size 0.7, 0.7
set style line   1 lt 1 lw 3 
set style line   2 lt 3 lw 3 
set style line   3 lt 2 lw 3 
set grid noxtics ytics
#set xtics ("4K" 4096, "16K" 16384, "64K" 65536, "128K" 131072,  "298992" 298992)
#set ytics ("2" 2, "5.0" 5,  "10" 10,  "20" 20, "40" 40, "80" 80, "100" 100) 
#set terminal png size 800 500
#set output "jaguar_ess.png"
set terminal postscript eps enhanced color "NimbusSanL-Regu" 
#fontfile "/usr/share/texmf-texlive/fonts/type1/urw/helvetic/uhvr8a.pfb" 14
set output "cpu-gpu-jyc-apoa1.eps"

plot "scale.dat"  using 1:2 title "node: 1GPU,16 CPU-cores" with linespoints ls 2 , "scale.dat" using 1:3 title "node:16 CPU-cores" with linespoints ls 1, "scale.dat" using 1:4 title "node:32 CPU-cores" with linespoints ls 3 

