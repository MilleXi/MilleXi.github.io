---
layout: post
title: "python安装与配置指南"
date:   2024-10-17
tags: [tech]
comments: true
author: MilleXi
---

一份给初学者的简单Python与VSCode安装&配置指南，五分钟轻松拿下！额外附赠Jupyter Notebook安装指南哦~

<!-- more -->

## 一、安装python

1. 指路[Python官方安装网站](https://www.python.org/downloads)

    <img src="https://millexi.github.io/images/16.png" alt="python1" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

2. 选择你想要的版本，点击下图中标出的download按钮进行下载

    <img src="https://millexi.github.io/images/17.png" alt="python2" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

    <font color=gray>这里笔者建议如无特殊需求不要下载最新版的python，可以选择次新版以避免一些比较抽象的版本兼容问题</font>

3. 下载完成后双击打开下载的文件，出现如下图画面

    <img src="https://millexi.github.io/images/18.png" alt="python3" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

    - 此处注意一定要**勾上**下面两个小勾 use和add……（上图红色框框处）

    - 接着点击**Install Now**

    - 等一会儿会提示下载完成，关闭页面，你已经装好python了，让我们来验证一下

4. 快捷键win+r，输入cmd，回车进入命令提示符，输入如下指令并回车

    ```bash
    python --version
    ```

    此时出现如下图，说明安装正确

    <img src="https://millexi.github.io/images/19.png" alt="python4" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

5. 接下来我们**给pip换源**，在刚刚的命令提示符页面中继续输入如下指令并回车

    ```bash
    python -m pip install --upgrade pip
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    ```

    如下图
    <img src="https://millexi.github.io/images/20.png" alt="python5" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

6. 测试一下是否可以正常使用pip安装一些python常用库

    *Python库可以理解为一组“现成的工具”，这些工具帮助你更轻松地完成各种任务，而不用从头开始写代码。就像你要做一个大餐，库就是帮你省事的“食材包”或者“厨房工具”。如果你想在电脑上实现一些功能，比如做数学运算、画图、访问网站，Python库里已经有专门为这些任务准备好的代码。你只需要“调用”这些库中的工具，就像打开包装、用现成的食材做菜一样，快速实现你想要的结果*

    - 继续在刚才的页面中输入如下指令并回车

    ```bash
    pip install jupyter
    ```

    显示如下图说明刚才的步骤全都被正确操作了

    <img src="https://millexi.github.io/images/21.png" alt="python6" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

    显示出如下图即说明Jupyter安装完成

    <img src="https://millexi.github.io/images/22.png" alt="python7" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

至此python就装好了😎

## 二、安装vscode

###### 因为本人比较喜欢vscode，轻量便捷，有很多插件

*VS Code（全名为Visual Studio Code）是一个轻量、免费的代码编辑器。它可以帮助你写代码、调试代码、以及运行程序。简单来说，VS Code就是一个“写代码的工具”。VS Code和Python的关系是，VS Code可以作为你编写Python程序的编辑器。它为Python开发者提供了许多便利的功能，比如代码提示、自动补全、错误检查，以及一键运行Python程序。通过安装Python扩展，VS Code还可以帮助你调试Python代码、管理虚拟环境，甚至直接在编辑器里运行Python脚本。*

1. 指路[VSCode官方安装网站](https://code.visualstudio.com/Download)

    <img src="https://millexi.github.io/images/23.png" alt="vscode1" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

2. 选择对应自己操作系统的版本进行下载，下载后双击打开文件，出现如下图

    <img src="https://millexi.github.io/images/24.png" alt="vscode2" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

    - 勾选我同意

    - 点击下一步，出现如下图

    <img src="https://millexi.github.io/images/25.png" alt="vscode3" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

    - 大胆勾选上上图所有勾勾，点击下一步

    - 再点击安装
    
    - 最后点完成
    
    - 让我们接下来测试一下你的安装成果

3. 桌面新建一个文件夹test，右键，点击显示更多选项，点击通过Code打开，进入vscode，如下图

    <img src="https://millexi.github.io/images/26.png" alt="vscode4" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

4. 点击右边导航栏中如上图中红色框内的按钮，出现如下图
    
    <img src="https://millexi.github.io/images/27.png" alt="vscode5" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

5. 在上图中用红框圈出的搜索栏内输入python并回车，出现下图，点击红框中的选项

    <img src="https://millexi.github.io/images/28.png" alt="vscode6" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

6. 点击下图中红框圈出的install按钮

    <img src="https://millexi.github.io/images/29.png" alt="vscode7" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

至此你的VSCode+Python基础已经配置好了😊

## 三、安装jupyter notebook插件（可选）

1. 与刚才一样在VSCode中搜索jupyter，出现下图，点击红框中的选项，并同样点击右侧install按钮

    <img src="https://millexi.github.io/images/30.png" alt="jupyter1" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

## 四、开始一个新代码文件！

1. 点开右侧导航栏中如下图红框中的按钮

    <img src="https://millexi.github.io/images/31.png" alt="test1" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

2. 戳开下图中这个红圈标出的箭头

    <img src="https://millexi.github.io/images/32.png" alt="test1" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

    此时出现如下图

    <img src="https://millexi.github.io/images/33.png" alt="test2" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

    TEST意味着你已经在这个文件夹下，现在让我们在里面创建一个代码文件

3. 右键TEST下方这个空白区域，点击弹窗中的New File按钮，如下图

    <img src="https://millexi.github.io/images/34.png" alt="test3" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

4. 如下图，输入test.py并回车

    <img src="https://millexi.github.io/images/35.png" alt="test4" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

5. 双击这个test.py进入代码编辑页面，如下图

    <img src="https://millexi.github.io/images/36.png" alt="test5" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

6. 在第一行中输入如下代码

    ```python
    print("hello world!")
    ```

    *注意(" ")这四个字符都必须是英文字符*

    如下图

    <img src="https://millexi.github.io/images/37.png" alt="test6" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

7. 点击上图中红圈所圈出的运行按钮，运行代码，出现如下图

    <img src="https://millexi.github.io/images/38.png" alt="test7" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

！恭喜踏入Python的世界 ~ ✨

## 五、开始一个新的Jupyter Notebook文件!（可选）

1. 同样在VSCode侧边文件夹内创建一个test_jupyter.ipynb文件
    
    <img src="https://millexi.github.io/images/39.png" alt="test8" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

2. 如下图中右上角这个select kernel按钮戳开

    <img src="https://millexi.github.io/images/40.png" alt="test9" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

3. 顶部出现如下图弹窗，点击python environments

    <img src="https://millexi.github.io/images/41.png" alt="test10" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

4. 出现下图，接着点击带五角星⭐的推荐选项

    <img src="https://millexi.github.io/images/42.png" alt="test11" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

5. 点击屏幕中间如下图的+code按钮

    <img src="https://millexi.github.io/images/43.png" alt="test12" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

6. 在弹出的代码框中输入如下代码

    ```python
    print("hello world!")
    ```

    如下图

    <img src="https://millexi.github.io/images/44.png" alt="test13" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

7. 戳一下框框旁边如下图的三角形运行按钮

    <img src="https://millexi.github.io/images/45.png" alt="test14" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

8. 运行完后出现如下图即成功

    <img src="https://millexi.github.io/images/46.png" alt="test15" style="display: block; margin: 0 auto;width: 95%; max-width: 800px; height: auto;">

至此你已经成功入门Jupyter Notebook啦！👍👍👍

欢迎继续学习更多的Python和Jupyter Notebook等的相关知识！让我们一起加油😄💞