## Anaconda安装 ##

-	Anaconda Linux 64位安装程序`Anaconda3-5.0.1-Linux-x86_64.sh`下载：

	>下载地址：[百度云盘](https://pan.baidu.com/s/1w90tNSWkDeb57w6NGpdumw)

	>提取码：`ysc6` 

-	Anaconda安装命令：

    	bash Anaconda3-5.0.1-Linux-x86_64.sh

-	安装完成之后需要重启`Terminal`终端，Anaconda才能生效。

-	在安装的过程中，会问你安装路径，直接回车`Enter`默认就可以了。有个地方问你是否将Anaconda安装路径加入到`bash`资源文件`.bashrc`中，输入`yes`，默认的是`no`。

-	如果没输入`yes`就需要配置环境变量，在`Terminal`终端输入以下命令使用`Gedit`文本编辑器打开`profile`文件：

		sudo gedit /etc/profile

-	在`profile`文件中添加以下语句，把语句中的`/home/bupt`换成你自己的Anaconda安装路径：

		export PATH=/home/bupt/anaconda3/bin:$PATH

-	`Ctrl`+`S`保存修改，然后退出`Gedit`。

-	重启`Terminal`终端，如果还是不行，则重启Linux系统。

-	配置好`PATH`后，可以通过以下命令检查配置是否正确：

		conda –version

-	可以通过以下命令查看Anaconda组件：

		conda list

-	Python版本：`3.6.3`

## TensorFlow安装 ##

-	安装好Anaconda后，运行以下命令安装`1.3.0`版本TensorFlow：

		conda install tensorflow==1.3.0