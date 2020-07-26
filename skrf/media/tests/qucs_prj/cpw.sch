<Qucs Schematic 0.0.19>
<Properties>
  <View=0,0,800,826,1,0,110>
  <Grid=10,10,1>
  <DataSet=cpw.dat>
  <DataDisplay=cpw.dpl>
  <OpenDisplay=1>
  <Script=cpw.m>
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
  <SUBST Subst1 1 470 420 -30 24 0 0 "12.9" 1 "100 mm" 1 "0.01 um" 1 "2e-4" 1 "0.022e-6" 1 "0.15e-6" 1>
  <CLIN CL1 1 290 170 -26 28 0 0 "Subst1" 1 "75e-3 mm" 1 "50e-6 mm" 1 "1 m" 1 "Air" 0 "yes" 0>
  <.SP SP1 1 100 390 0 73 0 0 "lin" 1 "1 GHz" 1 "10 GHz" 1 "11" 1 "no" 0 "1" 0 "2" 0 "no" 0 "no" 0>
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
