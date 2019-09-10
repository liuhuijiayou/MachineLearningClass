## Docker安装 ##

-	在线安装命令：

		sudo apt install docker.io

-	离线安装命令，使用`cd`命令进入对应版本目录后可输入以下命令安装：

		sudo dpkg -i *.deb

## Docker加速器 ##

-	使用在线安装方法时，可以先使用以下命令安装Docker加速器，以加快在线安装速度：

		sudo curl -sSL https://get.daocloud.io/daotools/set_mirror.sh | sh -s http://196340ca.m.daocloud.io

## Docker从文件载入镜像 ##

-	`openfaceallset.tar`镜像文件下载：

	>下载地址：[百度云盘](https://pan.baidu.com/s/1NLzLCBm1Ub9Hxw93t0cXkQ)

	>提取码：`5uow`

-	载入镜像命令：

		sudo docker load < openfaceallset.tar

## Docker运行命令 ##

-	添加映射命令：

		sudo xhost +local:root

-	开启容器命令：

		sudo docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /home/$USER:/home/$USER:rw -p 9000:9000 -p 8000:8000 -t -i openface/allset /bin/bash