#!/usr/bin/env wolframscript
(* ::Package:: *)

Run["sed -i 's/((//g' result.dat"]

Run["sed -i 's/), (/\\n/g' result.dat"]

Run["sed -i 's/))/\\n/g' result.dat"]

Run["sed -i 's/),)//g' result.dat"]

Run["sed -i 's/ //g' result.dat"]
