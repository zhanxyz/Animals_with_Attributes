------------------------------------------------------------------
Animals with Attributes Dataset, v1.0, May 22th 2009
------------------------------------------------------------------
Matlab code for Indirect Attribute Prediction (IAP) model
(C) May 22, Hannes Nickisch, <hn@tuebingen.mpg.de>
------------------------------------------------------------------

1) execute build_matfiles.m to obtain Matlab files from txt files containing 
   features (written to feat/x_<feat>_c<1-50>.mat.mat) and some 
   constants (written to constants.mat)

2) evaluate liblinear_cv5(cvsplit,log3_C) on the grid:
   log3_C = -13:-9
   cvsplit = 0:5, where 0 corresponds to the predefined train/test split 
                  and 1:5 are additional splits used for crossvalidation
   modify the path to the data and the path to liblinear
   
3) execute collect_results.m to see what you have learnt [1]
  
------------------------------------------------------------------
  
[1] should be similar to
split 0, C=6.27e-07: Acc = 27.379% (1692/6180)
split 0, C=1.88e-06: Acc = 27.557% (1703/6180)
split 0, C=5.65e-06: Acc = 26.990% (1668/6180)
split 0, C=1.69e-05: Acc = 26.100% (1613/6180)
split 0, C=5.08e-05: Acc = 25.162% (1555/6180)
split 1, C=6.27e-07: Acc = 27.780% (1689/6080)
split 1, C=1.88e-06: Acc = 27.829% (1692/6080)
split 1, C=5.65e-06: Acc = 27.730% (1686/6080)
split 1, C=1.69e-05: Acc = 26.974% (1640/6080)
split 1, C=5.08e-05: Acc = 26.069% (1585/6080)
split 2, C=6.27e-07: Acc = 29.859% (1670/5593)
split 2, C=1.88e-06: Acc = 30.324% (1696/5593)
split 2, C=5.65e-06: Acc = 31.128% (1741/5593)
split 2, C=1.69e-05: Acc = 30.771% (1721/5593)
split 2, C=5.08e-05: Acc = 29.555% (1653/5593)
split 3, C=6.27e-07: Acc = 19.648% (1150/5853)
split 3, C=1.88e-06: Acc = 19.341% (1132/5853)
split 3, C=5.65e-06: Acc = 18.469% (1081/5853)
split 3, C=1.69e-05: Acc = 17.564% (1028/5853)
split 3, C=5.08e-05: Acc = 16.778% (982/5853)
split 4, C=6.27e-07: Acc = 19.780% (1383/6992)
split 4, C=1.88e-06: Acc = 20.166% (1410/6992)
split 4, C=5.65e-06: Acc = 19.923% (1393/6992)
split 4, C=1.69e-05: Acc = 19.479% (1362/6992)
split 4, C=5.08e-05: Acc = 18.836% (1317/6992)
split 5, C=6.27e-07: Acc = 25.533% (1521/5957)
split 5, C=1.88e-06: Acc = 25.264% (1505/5957)
split 5, C=5.65e-06: Acc = 24.945% (1486/5957)
split 5, C=1.69e-05: Acc = 24.123% (1437/5957)
split 5, C=5.08e-05: Acc = 23.619% (1407/5957)
