#include <cstdio>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define DOWN_TO_MULTIPLE(x, y) (((x) / (y)) * (y))
#define ROUND_DIV(x, y) (((x) + ((y) / 2)) / (y))

int adaptive_quantize_(const int in_data, const int bitwidth) {
  int max_value;
  int val;

  max_value = (1 << bitwidth) - 1;

  int offset = in_data ? 0 : 1;
  int div = CEIL_DIV(in_data, max_value) + offset;
  // int div = 1 + CEIL_DIV(in_data, (max_value + 1));
  div--;
  div |= div >> 1;
  div |= div >> 2;
  div |= div >> 4;
  div |= div >> 8;
  div |= div >> 16;
  div++;

  printf("div: %d\n", div);
  val = ROUND_DIV(in_data, div) * div;
  return val;
}

int main() {
  int in_data;
  int bitwidth = 4;
  for (int i = 0; i < 256; i++) {
    in_data = i;
    int out_data = adaptive_quantize_(in_data, bitwidth);
    printf("%d -> %d\n", i, out_data);
  }
}