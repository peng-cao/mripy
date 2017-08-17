Author: Peter Kellman
Date: August 22, 2011

This folder contains sample datasets for image-space fat-water
separation and fat quantification. The datasets are as follows:

Zip file containing image data from 1 subject, 15 datasets, as
follows, where each mat-file consists of
data.images(row,col,slice,coil,echo), and data.TE (in seconds). 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

If these datasets are to be used for publication, the following
language can be used to address IRB compliance:

"The National Institutes of Health provided raw imaging data for the
purposes of technical development and evaluation of new methods for
magnetic resonance imaging (MRI). The raw datasets are
anonymized/unlinked and were determined to be 'Not Human Subjects
Research' under 45 CFR 46.101 (b) and 46.102 by the NIH Office of
Human Subjects Research Protections. This review is provided by OHSRP
in lieu of IRB review under the NIH FWA# 00005897."

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 



Each data.mat file also contains data.FieldStrength (all at 1.5T)
and data.PrecessionIsClockwise = 1, which signifies the clockwise precession
of main fat peak (-3.4ppm) relative to water.

All of these are coil combined. 
 
Sax corresponds to short axis cardiac, ch4 corresponds to the 4
chamber view, and cor is the coronal view to include both heart and
liver. Datasets10-15 which indicate "DBprep" incorporate a dark
blood prep (double IR) for better contrast of the myocardium so that
you can visualize the thin walled RV nicely. I included 3, 4, and 8
echo datasets and I can get others if you like (e.g., 2 echo).

The data quality was OK in terms of the fat water separation, although
there is some slight motion related ghosting on some. 

% Dataset1  sax GRE 4 echo

% Dataset2  ch4 GRE 4 echo

% Dataset3  sax GRE 3 echo

% Dataset4  ch4 GRE 3 echo

% Dataset5  sax GRE 8 echo

% Dataset6  ch4 GRE 8 echo

% Dataset7  cor GRE 3 echo

% Dataset8  cor GRE 4 echo

% Dataset9  cor GRE 8 echo

% Dataset10  sax GRE 4 echo DBprep

% Dataset11  ch4 GRE 4 echo DBprep

% Dataset12  sax GRE 3 echo DBprep

% Dataset13  ch4 GRE 3 echo DBprep

% Dataset14  ch4 GRE 8 echo DBprep

% Dataset15  sax GRE 8 echo DBprep

 
Dataset 16 & 17 are 2 echo (in and out of phase) for the same subject,
sax and 4 chamber cardiac, respectively. I did not get the DB prepped
version of these. They will be useful to those interested in 2
echoes. I did not get a coronal view for this case.

% Dataset16  sax GRE 2 echo

% Dataset17  ch4 GRE 2 echo

 
