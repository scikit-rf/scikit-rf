<Qucs Schematic 0.0.19>
<Properties>
  <View=0,-60,1406,800,0.751315,0,0>
  <Grid=10,10,1>
  <DataSet=diff_2xthru.dat>
  <DataDisplay=diff_2xthru.dpl>
  <OpenDisplay=1>
  <Script=diff_2xthru.m>
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
  <Pac P4 1 720 310 18 -26 0 1 "4" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <GND * 1 720 340 0 0 0 0>
  <GND * 1 720 200 0 0 0 0>
  <.SP SP1 1 240 490 0 65 0 0 "lin" 1 "10 MHz" 1 "10 GHz" 1 "1000" 1 "no" 0 "1" 0 "2" 0 "no" 0 "no" 0>
  <SUBST Subst1 1 510 530 -30 24 0 0 "4.413" 1 "1.51 mm" 1 "50 um" 1 "0.0182" 1 "1.712e-8" 1 "0.15e-6" 1>
  <MCOUPLED MS1 1 490 210 -30 88 0 0 "Subst1" 1 "1.76 mm" 1 "60 mm" 1 "1 mm" 1 "Kirschning" 1 "Kirschning" 1 "26.85" 0>
  <Eqn Eqn1 1 670 510 -30 16 0 0 "S11_dB=dB(S[1,1])" 1 "S41_dB=dB(S[4,1])" 1 "yes" 0>
  <Pac P3 1 720 170 18 -26 0 1 "3" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
</Components>
<Wires>
  <240 140 460 140 "" 0 0 0 "">
  <460 140 460 180 "" 0 0 0 "">
  <520 140 520 180 "" 0 0 0 "">
  <520 140 720 140 "" 0 0 0 "">
  <460 240 460 280 "" 0 0 0 "">
  <240 280 460 280 "" 0 0 0 "">
  <520 240 520 280 "" 0 0 0 "">
  <520 280 720 280 "" 0 0 0 "">
</Wires>
<Diagrams>
</Diagrams>
<Paintings>
</Paintings>
