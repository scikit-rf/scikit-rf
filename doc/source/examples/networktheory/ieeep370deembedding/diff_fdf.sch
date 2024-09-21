<Qucs Schematic 0.0.19>
<Properties>
  <View=0,-60,1553,800,0.751315,0,0>
  <Grid=10,10,1>
  <DataSet=diff_fdf.dat>
  <DataDisplay=diff_fdf.dpl>
  <OpenDisplay=1>
  <Script=diff_fdf.m>
  <RunScript=0>
  <showFrame=0>
  <FrameText0=Titre>
  <FrameText1=Auteur :>
  <FrameText2=Date :>
  <FrameText3=Version :>
</Properties>
<Symbol>
</Symbol>
<Components>
  <Pac P2 1 240 310 18 -26 0 1 "2" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <GND * 1 240 340 0 0 0 0>
  <Pac P1 1 240 170 18 -26 0 1 "1" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <GND * 1 240 200 0 0 0 0>
  <.SP SP1 1 240 490 0 65 0 0 "lin" 1 "10 MHz" 1 "10 GHz" 1 "1000" 1 "no" 0 "1" 0 "2" 0 "no" 0 "no" 0>
  <SUBST Subst1 1 510 530 -30 24 0 0 "4.413" 1 "1.51 mm" 1 "50 um" 1 "0.0182" 1 "1.712e-8" 1 "0.15e-6" 1>
  <MCOUPLED MS1 1 490 210 -30 88 0 0 "Subst1" 1 "2.05 mm" 1 "30 mm" 1 "1 mm" 1 "Kirschning" 1 "Kirschning" 1 "26.85" 0>
  <MCOUPLED MS2 1 670 210 -30 88 0 0 "Subst1" 1 "2.05 mm" 1 "20 mm" 1 "1 mm" 1 "Kirschning" 1 "Kirschning" 1 "26.85" 0>
  <MCOUPLED MS3 1 850 210 -30 88 0 0 "Subst1" 1 "6.15 mm" 1 "20 mm" 1 "1 mm" 1 "Kirschning" 1 "Kirschning" 1 "26.85" 0>
  <MCOUPLED MS4 1 1030 210 -30 88 0 0 "Subst1" 1 "2.05 mm" 1 "20 mm" 1 "1 mm" 1 "Kirschning" 1 "Kirschning" 1 "26.85" 0>
  <MCOUPLED MS5 1 1210 210 -30 88 0 0 "Subst1" 1 "2.05 mm" 1 "30 mm" 1 "1 mm" 1 "Kirschning" 1 "Kirschning" 1 "26.85" 0>
  <Pac P4 1 1440 310 18 -26 0 1 "4" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <GND * 1 1440 340 0 0 0 0>
  <Pac P3 1 1440 170 18 -26 0 1 "3" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <GND * 1 1440 200 0 0 0 0>
</Components>
<Wires>
  <240 140 460 140 "" 0 0 0 "">
  <460 140 460 180 "" 0 0 0 "">
  <460 240 460 280 "" 0 0 0 "">
  <240 280 460 280 "" 0 0 0 "">
  <520 180 640 180 "" 0 0 0 "">
  <700 180 820 180 "" 0 0 0 "">
  <520 240 640 240 "" 0 0 0 "">
  <700 240 820 240 "" 0 0 0 "">
  <1240 140 1240 180 "" 0 0 0 "">
  <1240 140 1440 140 "" 0 0 0 "">
  <1240 240 1240 280 "" 0 0 0 "">
  <1240 280 1440 280 "" 0 0 0 "">
  <880 180 1000 180 "" 0 0 0 "">
  <1060 180 1180 180 "" 0 0 0 "">
  <1060 240 1180 240 "" 0 0 0 "">
  <880 240 1000 240 "" 0 0 0 "">
</Wires>
<Diagrams>
</Diagrams>
<Paintings>
</Paintings>
