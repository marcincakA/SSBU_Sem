# ID Format Documentation

## Overview

The ID field in the dataset follows a specific format that encodes information about the sample reception date and patient order.

## Format Structure

The ID uses a 9-digit format: `YMMDDNNNN` where:

- **Y**: Last digit of the year from "prijem vzorky" (sample reception)
- **MM**: Month (01-12) of "prijem vzorky"
- **DD**: Day (01-31) of "prijem vzorky"
- **NNNN**: Order of patient based on reception time sequence

## Example

An ID of `312050001` would represent:
- `3`: Year ending in 3 (e.g., 2023)
- `12`: December
- `05`: 5th day of the month
- `0001`: First patient received on that day

## Notes

- Leading zeros are preserved as they contain important information
- The ID should always be handled as a string/text value, not a number
- Missing ID values should be left empty rather than filled with placeholder values
- This format allows for sequential ordering by date while maintaining a unique identifier for each patient 