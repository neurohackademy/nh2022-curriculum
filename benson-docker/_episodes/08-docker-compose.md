---
title: "Introduction to docker-compose"
teaching: 20
exercises: 0
questions:
- "How can I simplify the use of Docker containers for my users?"
- "How can synchronize multiple Docker containers together?"
objectives:
- "Be able to use a `docker-compose.yml` file to document and simplify the use of a container."
- "Understand how `docker-compose` can be used to start multiple containers that depend on each other."
keypoints:
- "The `docker-compose` command is intended to coordinate multiple containers that work together."
- "`docker-compose` can also document and automate the way a single container is invoked."
- "The `docker-compose` command reads its configuration from a `docker-compose.yml` file."
---

### Before we start...
The `docker-compose` tool is one of a number of tools for organizing containers, and it is evolving rapidly. In this lesson, we will focus on uses of `docker-compose` that arise commonly when developing reproducible computations. This is just one set of ways that `docker-compose` can be used, however. In industry, for example, `docker-compose` is often used to manage the elastic coordination of cloud computing nodes that host web-services.

### `docker-compose` can simplify the use of Docker containers
By this point, we've seen that the `docker` command is a powerful tool for interacting with containers and container images. It can be used to build containers, pull containers from DockerHub, and to run containers. We have also seen how a `Dockerfile` can encode a specific set of instructions that the `docker build` command can use to create a particular software environment in a container image. A frequent tactic for software projects is to include a `Dockerfile` in the project's GitHub repository root. This file generally creates a container image that is capable of supporting the software in the repository so that developers and users who download the software from GitHub can easily run it.

> ## docker-compose is an official part of docker...
> There are many tools for orchestrating and managing containers, but a
> particular advantage of `docker-compose` is that anyone who has Docker
> installed should be able to use `docker-compose` as well. This is a big
> advantage if your goal is to simplify the process of building and running a
> container.
{: .callout}

For example, suppose I am working on a team that is developing a Linux-based data analysis tool, but I only have a Windows machine. In order for me to test the code I've written, I will need to be able to run it on a Linux machine. A simple fix would be for us to include a Dockerfile that encodes all the software dependencies of the project on top of a Linux image such as the `alpine` Docker image we have used in previous lessons. Then, whenever I needed to test my code, I can use `docker run` to start the container and test my code.

Although this approach is very valuable, a frequent difficulty is that, in order to use software in this manner, one must be familiar enough with Docker to correctly run both the `docker build` and the `docker run` commands. These commands can be intimidating for inexperienced users, and for complex containers, they can themselves be very long.

One way to simplify the process of building and running a container for an end-user is to write a `docker-compose.yml` file. Such a file can be used to document the way that a container is intended to be built and run. In this lesson, we will demonstrate how to do this with the `alpine-sum` Docker image we created in an earlier lesson.


### Creating our first `docker-compose.yml` file
We will start by revisiting our `alpine-sum` `Dockerfile` from the [Advanced Containers]({{site.url}}/05b-advanced-containers/index.html) lesson. At the end of that lesson, we had a `Dockerfile` that looked like the following:

~~~
FROM alpine

COPY sum.py /home
# set script permissions
RUN chmod +x /home/sum.py
# add /home folder to the PATH
ENV PATH /home:$PATH

RUN apk add --update python3 py3-pip python3-dev

ENTRYPOINT ["python3", "/home/sum.py"]

CMD ["10", "11"]
~~~

In order to build this Docker image, we used the following commands:

~~~
$ docker build -t alpine-sum:v3 .
$ docker run alpine-sum:v3 1 2 3 4
~~~
{: .language-bash}

The `docker-compose` command uses a file called, predictably enough, `docker-compose.yml`. The `yml` at the end is sometimes `yaml` instead--both of these are endings for YAML files, which are text-based data files that use an intuitive format. Go ahead and use your favorite text editor to create a file `docker-compose.yml` with the following text:

~~~
version: '3'
services:
  alpine-sum:
    build: .
~~~

Then, rather than building **and** running the container, we can use a single command. If you still have the `alpine-sum` Docker image built locally on your machine, however, you might want to run `docker image rm alpine-sum` before running the following command. The `image rm` command will delete the previous `alpine-sum` image so that you will be able to see whether the following command rebuilds the image:

~~~
$ docker-compose run alpine-sum 1 2 3 4
~~~
{: .language-bash}
~~~
...
 ---> Running in a7a1806a67bb
Removing intermediate container a7a1806a67bb
 ---> d595137a5db8
Successfully built d595137a5db8
Successfully tagged dockerintro_alpine-sum:latest
WARNING: Image for service alpine-sum was built because it did not already exist. To rebuild this image you must use `docker-compose build` or `docker-compose up --build`.
sum = 10
~~~
{: .output}

Notice that if we run another command against the container, it won't be rebuilt.

~~~
$ docker-compose run alpine-sum 2 3 4 5
~~~
{: .language-bash}
~~~
sum = 14
~~~
{: .output}

### The `docker-compose.yml` file
Let's break down the `docker-compose.yml` file we just created to better understand it.

~~~
version: 3.0
~~~

The first line of the file simply states the version of the `docker-compose` schema that we're using. Note that this is the version of the syntax we are using in this file, not the version of `docker-compose` we are using.

~~~
services:
~~~

The `services` line is included in all `docker-compose.yml` files. It tells `docker-compose` which containers are managed by this file. Although our `docker-compose.yml` file has only one container, it is possible for there to be multiple containers, all of which get started up when you use `docker-compose`.

~~~
  alpine-sum:
~~~

The next line is indented in order to indicate that it is a sub-item under the `services` list (i.e., it is a service). We can give this service any name, but since we have been calling it `alpine-sum`, it is natural to keeep the same name.

~~~
    build: .
~~~

The final line is indented twice, indicating that it is part of the `alpine-sum` service. The `build` keyword indicates that the `alpine-sum` service should be built using the `docker build` command, and the `.` is simply an alias for the current directory (the directory containing the `docker-compose.yml` file). We could have alternately used the `image` tag to declare a public image from DockerHub such as with the line `    image: alpine`.


### Using `docker-compose` to run a server
We've focused in these lessons on using Docker to run single commands. However, recall that when we started the Jekyll server earlier, the Docker container stayed running in order to host the lesson web-page. This is a common use-case for Docker, and in order to manage such a container, we can use the `docker-compose up` command in place of `docker-compose run`.

To demonstrate this, let's return to the `docker-introduction` directory that we created earlier when we downloaded and unzipped the `docker-introduction` data.

~~~
$ cd docker-introduction-gh-pages
$ ls
~~~
{: .language-bash}
~~~
AUTHORS			_episodes		code
CITATION		_episodes_rmd		data
CODE_OF_CONDUCT.md	_extras			fig
CONTRIBUTING.md		_includes		files
LICENSE.md		_layouts		index.md
Makefile		aio.md			reference.md
README.md		assets			setup.md
_config.yml		bin
~~~
{: .output}

Previously, we used a somewhat long `docker run` command in order to start the Jekyll server here. This command (for Mac and Linux) was:
~~~
$ docker run --rm -it -v ${PWD}:/srv/jekyll -p 127.0.0.1:4000:4000 jekyll/jekyll:pages jekyll serve
~~~
{: .language-bash}

Let's create a `docker-compose.yml` file to manage this particular server. Each of the options in the above command line has a related keyword that can be used in the `docker-compose.yml` file, but we'll start with a simple YAML file.
~~~
version: '3'
services:
  my-jekyll-server:
    image: jekyll/jekyll:pages
~~~
{: .language-yaml}

This file is much like our earlier `docker-compose.yml` file, with the exception that it uses an `image:` tag to declare that the Docker image for the `my-jekyll-server` service should be started from the `jekyll/jekyll:pages` image obtained from DockerHub. In fact, we can go ahead and use this `docker-compose.yml` file, but since it only stores the name of the image, it won't simplify the command-line very much.

~~~
$ docker-compose run --rm -v ${PWD}:/srv/jekyll -p 127.0.0.1:4000:4000 my-jekyll-server jekyll serve
~~~
{: .language-bash}

(Don't forget that you can push control-C to exit out of the Jekyll-server container.) In the above line, we no longer need to specify the container name; however, we still have to specify everything else, including the name of the service (`my-jekyll-server`).

What would really be useful in this situation is if we could store some of the command-line options, like the port specification (`-p 127.0.0.1:4000:4000`) or the volumes specification `-v ${PWD}:/srv/jekyll` in the YAML file. Fortunately, we can! Docker compose supports a number of options for its services, including the `volumes:`, the `ports:`, and the `command:` keywords. Let's edit the `docker-compose.yml` file to include these.

~~~
version: '3'
services:
  my-jekyll-server:
    image: jekyll/jekyll:pages
    volumes:
      - "${PWD}:/srv/jekyll"
    ports:
      - "127.0.0.1:4000:4000"
    command: jekyll serve
~~~
{: .language-yaml}

> ## Why do some lines start with a dash?
> Notice in the above `docker-compose.yml` file, the `volumes:` and `ports:`
> keywords are followed by a line that starts with indentation then a `-`.
> This is because both of these keywords can accept lists--i.e., you can
> specify multiple volumes and multiple ports for each service. Each such
> entry begins with an indentation then `-`.
{: .callout}

Let's go ahead and test this out. Now that we have declared all of these details, we can start the Jekyll server with just the following command:

~~~
$ docker-compose up
~~~
{: .language-bash}
~~~
Creating dockerintroduction_my-jekyll-server_1 ... 
Creating dockerintroduction_my-jekyll-server_1 ... done
Attaching to dockerintroduction_my-jekyll-server_1
my-jekyll-server_1  | ruby 2.7.1p83 (2020-03-31 revision a0c7c23c9c) [x86_64-linux-musl]
my-jekyll-server_1  | Configuration file: /src/jekyll/_config.yml
my-jekyll-server_1  |             Source: /srv/jekyll
my-jekyll-server_1  |        Destination: /srv/jekyll/_site
my-jekyll-server_1  |  Incremental build: disabled. Enable with --incremental
my-jekyll-server_1  |       Generating... 
my-jekyll-server_1  |                     done in 1.048 seconds.
my-jekyll-server_1  |  Auto-regeneration: enabled for '/srv/jekyll'
my-jekyll-server_1  |     Server address: http://0.0.0.0:4000
my-jekyll-server_1  |   Server running... press ctrl-c to stop.
~~~
{: .output}

As you can hopefully see, this command successfully started the Jekyll server!
