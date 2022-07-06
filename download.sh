#!/bin/bash

TO_PATH=.
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-sisr-use-case/datasets/ucmerced-test-ds-pre-executed.tar.gz -O $TO_PATH/ucmerced-test-ds-pre-executed.tar.gz
tar xvzf $TO_PATH/ucmerced-test-ds-pre-executed.tar.gz -C $TO_PATH
rm $TO_PATH/ucmerced-test-ds-pre-executed.tar.gz
