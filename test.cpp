#include <iostream>

int main(void) {
  bool a[10];
  std::fill(a, a + 10, true);

  for (int i = 0; i < 10; i++) {
    std::cout << a[i] << std::endl;
  }
}