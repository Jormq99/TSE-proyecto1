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
> [!IMPORTANT]
> Es importante que la versión sea LTS y tenga soporte durante el desarrollo.

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

```vim
BB_HASHSERVE_UPSTREAM = "hashserv.yocto.io:8687"
SSTATE_MIRRORS ?= "file://.* https://sstate.yoctoproject.org/all/PATH;downloadfilename=PATH"
BB_SIGNATURE_HANDLER = "OEEquivHash"
BB_HASHSERVE = "auto"
```
Y agregar al final del archivo la indicación `IMAGE_FSTYPES += "wic.vmdk"` para que a la hora de que se cree la imagen, se genere un archivo con la extensión adecuada para poder correr la imágen en Virtual Box, con esto ya definido se puede proceder a generar la imágen, para ello se usa 

> [!NOTE]
> La extensión wic.vmdk, no es obligatoria en todos los casos, pero es requerida para este proyecto.

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
Para agregar recetas a la imagen se descarga del repositorio de kirkstone la rama de `meta-openembedded` que posee un conjunto de recetas y configuraciones prácticas para nuestra imagen, como lo puede ser agregar `python3`, lo que a su vez permite utilizar bibliotecas como `OpenCV` entre otras cosas, para esto debemos clonar el repo con el siguiente comando:
```bash
git clone -b kirkstone https://github.com/openembedded/meta-openembedded.git
```
Y una vez que se tiene esto, usamos el comando:
```bash
bitbake-layers add-layer meta-openembedded/meta-oe
bitbake-layers add-layer meta-openembedded/meta-python
```
Esto es de suma importancia para esta demostración, ya que nos permite agregar el `meta-oe` que a su vez nos permite usar herramientas como **_vim_** que son editores de texto o incluso **_ssh_** para servicios de red, al aplicar estos comandos, el archivo `bblayers.conf` se debería de ver como:
```vim
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
```vim
IMAGE_INSTALL:append = " \
                 python3-pip \
                 python3-pygobject \
                 python3-paramiko \
                 vim \
                 openssh \
                 opencv \
                "
```
> [!WARNING] 
> Si se agregan herramientas que no son parte de las recetas indicadas en los layers se va a generar un error, por lo que hay que asegurarse de que todo está incorporado de forma adecuada.

Estas utilidades se escogieron debido a la necesidad del procesamiento de imágenes y video, por lo que la prueba se realiza para verificar si existe algún conflicto con estas librerías y resolverlo antes de la implementación de los programas con **_OpenVino_**, con esto claro, se procede a la creación de imagen **_base_**

```bash
bitbake core-image-base
```
> [!NOTE]
> Al usar la imagen base se obtiene una mayor funcionalidad y la posibilidad de usar más recursos.


Importamos la imagen con el comando establecido [scp](#importar-imagen-a-escritorio-local).

Para finalizar probamos el comando vim y tambien la herramienta de python, primero creamos un archivo `.py` que se llame **_hola_** y agregamos la siguiente línea de codigo
```python
print("Imagen con python3 Primer Proyecto TSE - Jordani Mejía")
```
Salimos del editor de texto y podemos visualizar si funciona como debería usando `python3 hola.py`, esto nos debería arrojar algo como lo siguiente:

![image](https://github.com/aleguillen4/20231sTSE/assets/99856936/d23dd264-e96c-4d8c-b82f-576f754591b3)

## 3.Incluir archivos desde la creación de la imagen
Es de suma importancia la capcidad de generar la imagen con los archivos necesarios para el funcionamiento del proyecto desde su núcleo, ya que facilita la obtención de los archivos y su ejecución, para ello vamos a agregar un meta nuevo donde se van a encontrar todos lo recursos necesarios para esto, la forma de lograrlo es mediante los comandos

```bash
bitbake-layers create-layer meta-layername
bitbake-layers add-layer meta-layername
```

Para este ejemplo se usó el nombre de `meta-layersources` y dentro de esto se van a encontrar diversos archivos, pero hay una carpeta de suma importancia, que se llama **_recipes-example/example_** dentro de la cual vamos a generar un espacio donde agregar nuestros archivos, para ello ejecutamos

```bash
cd recipes-example/example
mkdir files
cd files
```
Una vez acá agregamos los elementos que sean necesarios, para este ejemplo solo se usa un archivo **_.py_** llamado `incluir.py`, este se ejecutará dentro de la imagen para mostrar como funciona, pero para ello debemos configurar el documento de `example_0.1.bb`, donde vamos a agregar su licencia y archivos necesarios
```vim 
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://COPYING.MIT;md5=3da9cfbcb788c80a0384361b4de20420"
SRC_URI += "file://incluir.py \ 
	   "
S = "${WORKDIR}"

do_install() {
	install -d ${D}${bindir}
	install -m 0755 incluir.py ${D}${bindir}
}
```
El código del `md5` se obtiene ejecutando este comando donde se encuentra dicho archivo de licencia
```bash
md5sum COPYING.MIT
```
Con esto podemos agregar la carpeta **_example_** al archivo de configuración `local.conf` y debería verse algo así 

```vim
IMAGE_INSTALL:append = " \
                 example \
                 python3-pip \
                 python3-pygobject \
                 python3-paramiko \
                 vim \
                 openssh \
                 opencv \
                "
```

Al agregar la carpeta ya se tiene configurada la creación de la imagen, para este caso se utiliza la versión **_core-image-x11-qemux86-64_**, de la siguiente forma

```bash
bitbake core-image-x11
```

Una vez importada y agregada la imagen al **_VirtualBox_** se navega por los archivos y se ejecuta el archivo `incluir.py`
```bash
cd ../../usr/bin
pyhton3 incluir.py
```
> [!NOTE]
> Al iniciar la imagen el directorio es /home/root pero esto puede variar, lo importante es la ubicación de nuestros archivos

El resultado se debe ver algo como esto 
![image](https://github.com/Jormq99/TSE-proyecto1/assets/99856936/f1d7002c-7bbd-41dc-a211-d370bc6ca071)

