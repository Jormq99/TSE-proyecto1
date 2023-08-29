# Bitácora - Primer Proyecto Sistemas Embebidos
## 1.Creacción de la imagen mínima con Yocto Project
Yocto es una herramienta que nos permite crear imágenes a la medida mediante el sistema modular de Linux, para esto usamos el componente Poky, que contienen toda la información necesaria para poder llevar a cabo el desarrollo de dichas imágenes, por lo que debemos instalar Yocto, junto con algunas de sus dependencias y clonar el repositorio para poder iniciar, para ello debemos usar: 
```bash
sudo apt install gawk wget git diffstat unzip texinfo gcc build-essential chrpath socat cpio python3 python3-pip python3-pexpect xz-utils debianutils iputils-ping python3-git python3-jinja2 libegl1-mesa libsdl1.2-dev pylint3 xterm python3-subunit mesa-common-dev zstd liblz4-tool
git clone git://git.yoctoproject.org/poky
```
Una vez tenemos el repo en nuestra carpeta debemos acceder a él con el comando `cd poky`, una vez tenemos esto, debemos definir con cual versión de yocto vamos a trabajar, en este caso se usa la versión **4.0 Kirkstone**, por lo que se usan los siguientes comandos
```bash
git checkout -t origin/kirkstone -b my-kirkstone
git pull
```
Con esto ya podemos cargar los recursos y emepezar a configurar la imagen mínima, para cargar estos recursos se debe usar
```bash
source oe-init-build-env
```
Y una vez se inicie el entorno se dirige a la ubicación de configuración mediente:

```bash
cd build/conf
vim local.conf
```
Este archivo de `local.conf` contiene la información correspondiente a las características que esperemos que tenga nuestra imagen, dentro de los primeros pasos debemos asegurarnos de descomentar las siguientes líneas del archivo, para acelerar el proceso de construcción

```
BB_HASHSERVE_UPSTREAM = "hashserv.yocto.io:8687"
SSTATE_MIRRORS ?= "file://.* https://sstate.yoctoproject.org/all/PATH;downloadfilename=PATH"
BB_SIGNATURE_HANDLER = "OEEquivHash"
BB_HASHSERVE = "auto"
```
Y agregar al final del archivo la indicación `IMAGE_FSTYPES += "wic.vmdk"` para que a la hora de que se cree la imagen, se genere un archivo con la extensión adecuada para poder correr la imágen en Virtual Box, con esto ya definido se puede proceder a generar la imágen, para ello se usa 

```bash
bitbake -k core-image-minimal
```

Y una vez finalizado el proceso se exporta a la computadura local mediente el comando `scp` en la consola **cmd** con las siguiente indicación

```bash
scp jordanimejia@172.176.181.48:/home/jordanimejia/yocto/poky/build/tmp/deploy/images/qemux86-64/core-image-minimal-qemux86-64.wic.vmdk "D:\Taller_Emb"
```

Con esta imagen mínima agregada a la herramienta Virtual Box, se debería generar algo como esto 
![imagen_minima](https://github.com/Jormq99/TSE-proyecto1/assets/99856936/9f30429d-6537-4b8b-9209-075a3b1c9d6e)

El user `root` es generado por defecto y permite llevar a cabo tareas administrativas con credenciales de súper usuario, pero de momento solo se ingresó para la demostración de la funcionalidad de la imagen generada.

## 2.Agregar recetas a mi imagen 
### "Poner como se agregraron las herramientas como python, editor vim u otros a la imagen"
