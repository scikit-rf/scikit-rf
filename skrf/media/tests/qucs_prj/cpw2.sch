<Qucs Schematic 0.0.19>
<Properties>
  <View=0,0,844,826,1,0,0>
  <Grid=10,10,1>
  <DataSet=cpw2.dat>
  <DataDisplay=cpw2.dpl>
  <OpenDisplay=1>
  <Script=cpw2.m>
  <RunScript=0>
  <showFrame=0>
  <FrameText0=Title>
  <FrameText1=Drawn By:>
  <FrameText2=Date:>
  <FrameText3=Revision:>
</Properties>
<Symbol>
</Symbol>
<Components>
  <GND * 1 120 250 0 0 0 0>
  <Pac P1 1 120 210 18 -26 0 1 "1" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <Pac P2 1 450 230 18 -26 0 1 "2" 1 "50 Ohm" 1 "0 dBm" 0 "1 GHz" 0 "26.85" 0>
  <GND * 1 450 260 0 0 0 0>
  <.SP SP1 1 100 390 0 73 0 0 "lin" 1 "0.2 GHz" 1 "20 GHz" 1 "100" 1 "no" 0 "1" 0 "2" 0 "no" 0 "no" 0>
  <SUBST Subst1 1 470 420 -30 24 0 0 "4.5" 1 "1.55 mm" 1 "35 um" 1 "0.018" 1 "1.7e-8" 1 "0.15e-6 um" 1>
  <Eqn Eqn1 1 690 430 -30 16 0 0 "S11_dB=dB(S[1,1])" 1 "S11_phi=arg(S[1,1])" 1 "S21_dB=dB(S[2,1])" 1 "S21_phi=arg(S[2,1])" 1 "yes" 0>
  <CLIN CL1 1 290 170 -26 28 0 0 "Subst1" 1 "3.00 mm" 1 "0.30 mm" 1 "25 mm" 1 "Metal" 1 "no" 1>
</Components>
<Wires>
  <120 240 120 250 "" 0 0 0 "">
  <450 170 450 200 "" 0 0 0 "">
  <320 170 450 170 "" 0 0 0 "">
  <120 170 120 180 "" 0 0 0 "">
  <120 170 260 170 "" 0 0 0 "">
</Wires>
<Diagrams>
</Diagrams>
<Paintings>
</Paintings>
