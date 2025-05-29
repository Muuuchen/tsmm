## install intel mkl
install with https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&linux-install=apt

```
apt update
apt install -y gpg-agent wget
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor |  tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" |  tee /etc/apt/sources.list.d/oneAPI.list
apt update
apt install intel-oneapi-mkl
```


```
git clone git@github.com:Muuuchen/tsmm.gitapt install -y gpg-agent wget
cmake -B build
cmake --build build -- -j4

```
