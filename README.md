![Shark](./img/shark.jpg?raw=true)


# JAWS: Justified Automated Weather Station
___

## About

JAWS is a scientiﬁc software workﬂow to ingest Level 2 (L2) data in the multiple formats now distributed, harmonize it into a common format, and deliver value-added Level3 (L3) output suitable for distribution by the network operator, analysis by the researcher, and curation by the data center. 

Automated Weather Station (AWS) and AWS-like networks are the primary source of surface-level meteorological data in remote polar regions. These networks have developed organically and independently, and deliver data to researchers in idiosyncratic ASCII formats that hinder automated processing and intercomparison among networks. Moreover, station tilt causes signiﬁcant biases in polar AWS measurements of radiation and wind direction. Researchers, network operators, and data centers would beneﬁt from AWS-like data in a common format, amenable to automated analysis, and adjusted for known biases.

The immediate target recipient elements are polar AWS network managers, users, and data distributors. L2 borehole data suffers from similar interoperability issues, as does non-polar AWS data. Hence our L3 format will be extensible to global AWS and permafrost networks. JAWS will increase *insitu* data accessibility and utility, and enable new derived products.

## Overview

### JAWS consists of:

#### 1) Standardization

Convert L2 data (usually ASCII tables) into a netCDF-based L3 format compliant with metadata conventions (Climate-Forecast and ACDD) that promote automated discovery and
analysis. 

#### 2) Adjustment

Include value-added L3 features like the Retrospective, Iterative, Geometry-Based (RIGB) tilt angle and direction corrections, solar angles, and standardized quality flags. 

#### 3) API

Provide a scriptable API to extend the initial L2-to-L3 conversion to newer AWS-like networks and instruments.

___
## Installation
### Linux/unix/win
#### Requirements:
* writable directory
* anaconda/miniconda
* installed unzip package

From within a writable directory, run the following command:
``` html
$ conda install -c conda-forge jaws
```

## Running JAWS
### Example
Translating L2 ASCII formats into homogenized netCDF format:
``` html
$ jaws --L2=gcnet L2.ascii L3.nc
```
Storing AWS-like data using DSG convention:
``` html
$ jaws --L2=gcnet --featureType L2.ascii L3.nc
```
Unit-test to verify data:
``` html
$ jaws --L2=gcnet --kelvin sample_L2.ascii sample_L3.nc
```
Annotate L2b netCDF with CF and ACDD variable and global metadata:
``` html
$ jaws --L2=imau --creator_email=’janedoe@summit.com’ --L2.ascii L3.nc
```
Derive value-added data and metadata:
``` html
$ jaws --L2=gcnet --solar_zenith_angles L2.ascii L3.nc
```
<!--
* Should see similar log output to the following:
``` html
DEBUG : filemanager     Creating packet: './Example_data/AKUL232'
DEBUG : filemanager     Dropping packet: './Example_data/AKUL232'
DEBUG : filemanager     Creating packet: './Example_data/AKUL232'
DEBUG : filemanager     Creating packet: 1
DEBUG : filemanager     Creating packet: './Example_data/AKUL232.log'
DEBUG : filemanager     Creating packet: './Example_data/AUPA299'
DEBUG : filemanager     Dropping packet: './Example_data/AUPA299'
DEBUG : filemanager     Creating packet: './Example_data/AUPA299'
DEBUG : filemanager     Creating packet: 2
```
-->

___
## Credit

This software is being developed by the University of California Irvine under NASA Advanced Information Systems Technology (AIST) Proposal and Project 80NSSC17K0540.



## Resources

* [API Reference](https://github.com/jaws/jaws/blob/master/API.md)
* [Release Notes](https://github.com/jaws/jaws/releases)
* [Gallery](https://github.com/jaws/jaws/wiki/Gallery)
* [Examples](https://)


## Full Documentation

See the [Wiki](https://github.com/jaws/jaws/wiki/) for full documentation, examples, operational details and other information.


## Bugs and Feedback

For bugs, questions and discussions please use the [GitHub Issues](https://github.com/jaws/jaws/issues).

 
## Copyright and License

