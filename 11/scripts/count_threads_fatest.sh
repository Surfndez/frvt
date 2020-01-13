#!/bin/bash

watch -n 30 "top -n1 | grep fa_test | wc -l"
