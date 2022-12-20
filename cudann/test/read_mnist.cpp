#include <iostream>
#include <vector>
#include <fstream>
#include <string>

int reverse_int (int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void read_MNIST(int n_images, int n_features, std::string filename, std::vector<std::vector<double>> &arr) {
    arr.resize(n_images, std::vector<double>(n_features));
    std::ifstream file (filename, std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);

        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows = reverse_int(n_rows);

        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols = reverse_int(n_cols);

        for(int i=0;i<number_of_images;++i) {
            for(int r=0;r<n_rows;++r) {
                for(int c=0;c<n_cols;++c) {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
}
 
int main() {
  std::vector<std::vector<double>> data;

  const int n_images = 10000;
  const int n_features = 28*28;
  const std::string filename = "C:\\t10k-images.idx3-ubyte";

  read_MNIST(n_images, n_features, filename, data);
 
  return 0;
}