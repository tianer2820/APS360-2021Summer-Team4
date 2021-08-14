# UofT 2021 Summer APS360 Team4 Final Project

This repo contains all the codes/model used during the team project of Team 4.

## Overview of the project

This project aims to produce an image translation neural network model that converts real life photos to pixel arts.

This repo includes the codes of the final model, the baseline model used as comparision, and any utility codes used to prepare datasets and produce demo images. The dataset and model weight used are not included due to the filesize limit.

## Background

Pixel art is an popular art form in the game industry. Since pixel art cannot be simply produced by interpolation or 3D rendering, large background and animations would be time consuming to produce. Thus we want to speed up this process with machine learning. Artists should be able to use the generated images as drafts and improve base on them.

Our task is style-transfering a real photo into a pixel art. Previous works in this area includes Deep-style-transfer, pix2pix network, cycle-gan, UNIT, and many other. Our final model is a modified version of the cycle gan.

## Project structure
- **baseline_model**: the original "deep style transfer" code, includes some demo results
- **baseline_model_pix2pix**: a modified pix2pix model that accepts unpaired dataset. Also include dataset preparing codes
- **demo**: demo codes for both baseline and our primary model. Also contains some demo results
- **eval_code**: code used to evaluate our final model
- **src**: codes for our primary model
