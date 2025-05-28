## install mkl
```shell
apt update
apt install -y gpg-agent wget
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor |  tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" |  tee /etc/apt/sources.list.d/oneAPI.list
apt update
apt install intel-oneapi-mkl




Create the YUM repository file in the /temp directory as a normal user.
```shell
tee > /tmp/oneAPI.repo << EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
 ```
 Move the newly created oneAPI.repo file to the YUM configuration directory /etc/yum.repos.d.

```shell
sudo mv /tmp/oneAPI.repo /etc/yum.repos.d
```
```shell
yum install intel-oneapi-mkl-devel
```


```shell 
cmake -B build
make --build build
```
