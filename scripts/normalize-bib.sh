#!/bin/bash

set -eo pipefail

for input in "$@"; do
    biber --tool \
        --isbn-normalise \
        --output-align \
        --output-encoding=ascii \
        --output-field-replace=location:address,journaltitle:journal \
        --output-fieldcase=lower \
        --output-format=bibtex \
        --output-indent=2 \
        --output-legacy-dates \
        --output-file >(sponge "$input") \
        "$input"
done
