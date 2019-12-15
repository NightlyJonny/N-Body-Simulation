set terminal pngcairo size 800, 600 enhanced font "Calibri, 12"
set output "MassDistribution??.png"
set key box top right
set title "Mass Distribution at E=??" font "Calibri, 18"
set xlabel "Mass"
set ylabel "Count"
set mxtics
set mytics
set grid xtics ytics mxtics mytics lt -1 lt rgb "grey" lw 2, lt 0
