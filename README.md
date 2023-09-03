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
### Importar imagen a escritorio local
```bash
scp jordanimejia@172.176.181.48:/home/jordanimejia/yocto/poky/build/tmp/deploy/images/qemux86-64/core-image-minimal-qemux86-64.wic.vmdk "D:\Taller_Emb"
```

Con esta imagen mínima agregada a la herramienta Virtual Box, se debería generar algo como esto 
![imagen_minima](https://github.com/Jormq99/TSE-proyecto1/assets/99856936/9f30429d-6537-4b8b-9209-075a3b1c9d6e)

El user `root` es generado por defecto y permite llevar a cabo tareas administrativas con credenciales de súper usuario, pero de momento solo se ingresó para la demostración de la funcionalidad de la imagen generada.

## 2.Agregar recetas a mi imagen 
Para agregar recetas a la imagen se descraga del repositorio de kirkstone la rama de `meta-openembedded` que posee un conjunto de recetas y configuraciones prácticas para nuestra imagen, como lo puede ser agregar `python3`, lo que a su vez permite utilizar bibliotecas como `OpenCV` entre otras cosas, par esto debemos clonar el repo con el siguiente comando:
```bash
git clone -b kirkstone https://github.com/openembedded/meta-openembedded.git
```
Y una vez que se tiene esto, usamos el comando:
```bash
bitbake-layers add-layer meta-openembedded/meta-oe
bitbake-layers add-layer meta-openembedded/meta-python
```
Esto es de suma importancia para esta demostración, ya que nos permite agregar el `meta-oe` que a su vez nos permite usar herramientas como **vim** que son editores de texto o incluso **ssh** para servicios de red, al aplicar estos comandos, el archivo `bblayers.conf` se debería de ver como:
```
# POKY_BBLAYERS_CONF_VERSION is increased each time build/conf/bblayers.conf
# changes incompatibly
POKY_BBLAYERS_CONF_VERSION = "2"

BBPATH = "${TOPDIR}"
BBFILES ?= ""

BBLAYERS ?= " \
  /home/jordanimejia/yocto/poky/meta \
  /home/jordanimejia/yocto/poky/meta-poky \
  /home/jordanimejia/yocto/poky/meta-yocto-bsp \
  /home/jordanimejia/yocto/poky/build/meta-openembedded/meta-oe \
  /home/jordanimejia/yocto/poky/build/meta-openembedded/meta-python \
  "
```
Para finalizar se debe configurar el archivo `local.conf` donde vamos a indicarle que debe instalar de las recetas que agregamos
```
IMAGE_INSTALL:append = " \
                 python3-pip \
                 python3-pygobject \
                 python3-paramiko \
                 vim \
                 openssh \
                 opencv \
                "
```

Estas utilidades se escogieron debido a la necesidad del procesamiento de imágenes y video, por lo que la prueba se realiza para verificar si existe algún conflicto con estas librerías y resolverlo antes de la implementación de los programas con **OpenVino**, con esto claro, se procede a la creación de imagen **base**, ya no utilizamos la mínima

```bash
bitbake core-image-base
```

Importamos la imagen con el comando establecido [scp](#importar-imagen-a-escritorio-local).

Para finalizar probamos el comando vim y tambien la herramienta de python, primero creamos un archivo `.py` que se llame **hola** y agregamos la siguiente línea de codigo
```python
print("Imagen con python3 Primer Proyecto TSE - Jordani Mejía")
```
Salimos del editor de texto y podemos visualizar si funciona como debería usando `python3 hola.py`, esto nos debería arrojar algo como lo siguiente:
![image](https://github.com/aleguillen4/20231sTSE/assets/99856936/d23dd264-e96c-4d8c-b82f-576f754591b3)
