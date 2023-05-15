# Tutorial 1: Setting up your AWS Account

Aloha challenge participant 🤙 We are glad to welcome you to the fifth Global Data Science Challenge (GDSC). This year is all about [fighting river blindness](https://gdsc.ce.capgemini.com/app/data). On this journey, we will be using a few different machine learning techniques and algorithms as well as the AWS cloud ☁️. This very first tutorial is about helping you setting up your AWS account. In this tutorial, we will:

- Set up the AWS Sagemaker Studio environment which we will use throughout the challenge,
- Add a custom docker image to the environment, which has all preliminary libraries installed,
- Download the tutorials (Jupyter Notebooks) for the challenge,
- Learn how to work with Amazon Sagemaker Studio Notebooks, and
- Set up auto shutdown for idle instances to save costs.

Along with this tutorial, there is a [video](https://web.microsoftstream.com/video/ce2af85f-e8b3-4445-ba43-2331baddb64a) which walks you through the steps performed here. The video is optional for you. In this notebook tutorial the command line interface (CLI) is used for the majority of the tasks.  In the video, we mainly use the AWS Web Interface (also called AWS Console) to achieve the same. This is sometimes a bit easier, especially if you don't have a tech background. Feel free to choose whichever you prefer. 

Since your group only need to perform the tasks in this tutorial once, only one of your team members need to go through this tutorial. Please ask your team mates first!

- Not finished? Alright, lets jump in!
- Finished? Nice! We still recommend to go through section *6. Work with Amazon Sagemaker Studio Notebooks* and maybe take a look at section *7. Optional references*. 


## Setting up the Sagemaker Studio images and kernels

During the challenge, we will be working in [Sagemaker Studio](https://aws.amazon.com/sagemaker/studio/). This service provides a managed [jupyter lab](https://jupyter.org/) environment in which we can run our code. To use Sagemaker Studio we need to create a sagemaker domain and add a user to this domain (one per team is sufficient). Afterwards, we will download the training notebooks and code to the freshly created jupyter lab. In order to execute the notebooks, there are plenty dependencies and libraries we need to have installed. To simplify this for you, we prepared a custom Docker image that has everything already installed and set up. We will just need to attach this image to your sagemaker studio domain and you can use it afterwards as a kernel in juypter lab.

TLDR; In this section of the tutorial, we will:
- Create a new sagemaker domain,
- Create a new sagemaker user in that domain,
- Attach the custom docker image to the domain, 
- Download the notebooks to sagemaker, and
- Start a new notebook with the kernel from our custom docker image. 

You can either do this manually (shown in the corresponding [video](https://web.microsoftstream.com/video/ce2af85f-e8b3-4445-ba43-2331baddb64a) for this Tutorial) or you can follow the steps below to set up the environment via the aws cloud shell.

Lets jump in!



### 1.  Starting AWS Cloudshell


Open the AWS Management Console, navigate to your region (us-east-1) and start the [AWS Cloud Shell](https://docs.aws.amazon.com/cloudshell/latest/userguide/welcome.html).

![](docs/start.png "Start AWS Cloud Shell")

A new window opened which looks like this:

![](docs/cloudshell.png "First page of AWS Cloud Shell")


### 2. Set up Sagemaker Studio from the AWS Cloud Shell

In the aws cloud shell, enter the following commands:

- This will download the preliminary files to attach the custom docker image to the sagemaker domain:

```bash
git clone https://gdsc-code-commit-user-at-954362353459:6intZSiLZxXrdboGMDDSvf9VvnpiPOo+0JoiYxZBSq4=@git-codecommit.eu-central-1.amazonaws.com/v1/repos/gdsc5-tutorials-public
cd gdsc5-tutorials-public
```

- Now its time to create a new Sagemaker domain. <br>
    Before we can do this, we need to retrieve a few values from the AWS account and set them as variables in our current session. In particular, we need to know the Virtual Private Cloud (VPC) ID and the subnets which are attached to this VPC. Our sagemaker domain will be placed in this VPC. You don't need to know all about VPCs and subnets. For now its enough to know that this prevents the the outside world from accessing our resources and allows us to connect to the internet and other AWS resources. We use the default VPC which is pre-created by AWS. 

```bash
VPCID=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
SUBNETID0=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPCID --query 'Subnets[0].SubnetId' --output text)
SUBNETID1=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPCID --query 'Subnets[1].SubnetId' --output text)
```

- To use the domain, we also need an Identity and Access Management (IAM) role. This role will be attached to the Sagemaker domain. It allows us to use other AWS services (such as downloading data from S3) from our Sagemaker Studio. Lets create the role and set its ID as a variable in the session.

```bash
SMROLEARN=$(aws iam create-role --role-name sagemaker-exec-role --assume-role-policy-document file://scripts/create-domain/assume-role-policy.json --query 'Role.Arn')
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess --role-name sagemaker-exec-role
aws iam put-role-policy --role-name sagemaker-exec-role --policy-name sm-execution-policy --policy-document file://scripts/create-domain/inline-policy.json 
```

- With all these variables in place, we are ready to create a new Sagemaker domain, called gdsc-domain, and a new user profile, called team-profile. (You can change these values if you want):

# TODO: when creating a domain can we disable Canvas, Forecast, and Projects? (we can in the console!)
```bash
aws sagemaker create-domain --auth-mode IAM --domain-name gdsc-domain --subnet-ids $SUBNETID0 $SUBNETID1 --vpc-id $VPCID --default-user-settings="{\"ExecutionRole\":$SMROLEARN }"
DOMAINID=$(aws sagemaker list-domains --query 'Domains[0].DomainId' --output text)
```
- The domain needs a few minutes (approx. 5 - 7 min) to configure. So you can grep yourself a nice warm beverage of your choice ☕. Afterwards, run this (if it fails, wait for 2-3 min and try to rerun it pls.):

```
aws sagemaker create-user-profile --domain-id $DOMAINID --user-profile-name team-profile
```

- Lastly, we want to run this command in the cloud shell in order to attach the custom image to the previously created domain

# TODO: replace this with most recent script for adding new image

```bash
cd scripts/attach-image
make create-image REGION=us-east-1
```

The image will take a few minutes until it is visible in the studio. We will take a look at it later on.

Did all of the commands above execute successfully? Awesome! Lets open the Sagemaker Studio! If not, I recommend you watch the video for this tutorial.


### 3. Open the Sagemaker Studio domain 

We can find the created Sagemaker Studio Domain in the AWS Management Console. First, navigate to the AWS Sagemaker Service. The easiest way to get there is via the search bar:

![](docs/searchbar.png "Search bar in AWS Management Console")

Second, open the Sagemaker Studio by clicking on "Studio" in the control panel and afterwards on the orange button "Sagemaker Studio".

![](docs/smstudio.png "Sagemaker Studio Button")

Now, you should be able to see the Users overview and a profile called "team-profile". To open the Sagemaker Studio, click on "Launch app" and, in the dropdown, on "Studio".

![](docs/user-profiles.png "Overview of the user-profiles")

This will open the Sagemaker Studio (it looks like and essentially is Jupyter Lab) after 4-5 minutes loading time. It should look like this, after it loaded completely:

![](docs/vanilla-smstudio.png "A fresh sagemaker studio")

### 4. Clone the notebooks into Sagemaker Studio

In the left pane, click on the "Git" sign second in the list. Afterwards, click on "Clone a repository" and a pop up opens.

![](docs/smstudio-git.png "The git sign and clone repository button in Sagemaker Studio")

In the pop up, enter the repository URL and click on "CLONE": 

# TODO: replace this with new tutorials from Github (public repo)! 

```
https://gdsc-code-commit-user-at-954362353459:6intZSiLZxXrdboGMDDSvf9VvnpiPOo+0JoiYxZBSq4=@git-codecommit.eu-central-1.amazonaws.com/v1/repos/gdsc5-tutorials-public
```

![](docs/clone-pop-up.png "Clone pop up")


This will add all scripts and tutorials to the directory gdsc5-tutorials-public. To view it, navigate back to the file system overview (folder sign in the left pane, first in the list). Here double click on the directory to open it.


![](docs/sm-filesystem.png "File system overview in Sagemaker Studio")

### 5. Set up version control


Now we need to replace the remote from the previously downloaded git repository, so that we can use our own [CodeCommit](https://www.amazonaws.cn/en/codecommit/#:~:text=Amazon%20CodeCommit%20is%20a%20fully,worry%20about%20scaling%20its%20infrastructure.) repository. CodeCommit a version control management system from AWS, just like Github or Gitlab. We will create a repository to store our code and add this as our new remote, i.e. the safe location where our code is pushed to after we changed it locally. We will show you how all this works!

To start, we need to open a system terminal from the launcher. If the launcher didn't open automatically right from the start, click on the little plus sign at the top left corner, right below "Amazon Sagemaker Studio". Now we can click on "System Terminal" in the section "Utilities and files" and a new Terminal opens.

![](docs/image-terminal.png "How to open the Launcher and the image terminal")

In the terminal, enter the following commands (you can change the repository name (set to 'gdsc5-repo') as you wish)

```
REPO_URL=$(aws codecommit create-repository --repository-name gdsc5-repo --query 'repositoryMetadata.cloneUrlHttp' --output text)
git config --global credential.helper '!aws codecommit credential-helper $@'
git config --global credential.UseHttpPath true
cd gdsc5-tutorials-public/
git remote remove origin
git remote add origin $REPO_URL
git push --set-upstream origin master 
```

The past few comments created a repository in your team's AWS account and configured your local repository to use your CodeCommit repo as a remote location. If you pull or push in the future it will always refer to the repository in your AWS account. Cool, right? 😎    You can now interact with your repository from the terminal (as we just did) or from Git tab in the left pane in Sagemaker Studio.

FYI: if you need more information about git and why you should use it, take a look at [this](https://git-scm.com/).



### 6. Work with Amazon Sagemaker Studio Notebooks

Our Sagemaker Studio is now ready for development. In this part, we want to show you how to work with the Studio environment. In particular, we will learn how to:

* Open a new/existing notebook and start a jupyter kernel,
* Change to another kernel,
* Change the instance type for our notebook, and
* Stop an instance.

All these tasks will be performed in the Sagemaker Studio UI. Yes you heard right, no more CLI! 🥳

- First we will learn how to open a Jupyter Notebook. You can either start a new notebook from the launcher or by just opening one from the file system (little folder icon in the left pane) by clicking on it. As this is straight forward, we will only show you how to create a new one from the launcher. 

- In the launcher under "Notebooks and compute resources" choose a SageMaker image. Our image is called 'gdsc5-smstudio-custom'. This is btw the image that we attached earlier. If you cant see it, there might have been a problem while adding it. You can refer to the video to see how we do it via the AWS Management Console.

![](docs/choose-kernel.png)

- After you selected the gdsc image click on Python Notebook to open a notebook.

![](docs/create-notebook.png)

- Now the kernel starts, which takes some time. You should wait around 5 - 10 minutes (the image is pretty huge and loading it takes some time). After it runs, the top right bar should look like this

![](docs/running-kernel.png)

- If you chose the wrong kernel or want to try another one, you can either change the selection in the launcher or switch the kernel of a running notebook. To do the latter, click on the kernel name in the notebook and select another image. 

![](docs/switch-kernel.png)

- When the kernel has started, it will probably have chosen a default instance. An istance refers to an virtual machine. The instance types differ by their hardware capacities, e.g. how much memory they have or whether they have a GPU, which determines how much an hour of compute cost. All are paid the their they are running (including idle time). The instances to choose from are limited for the challenge. You can only select three:

    - ml.t3.medium (default): The cheapest and weakest. 2 vCPU and 4 GiB RAM
    - ml.m5.large: The highest performance without a GPU. Medium price. 2 vCPU and 8 GiB RAM
    - ml.g4dn.xlarge (GPU): Accelerated Computing. Very expensive. 4 vCPU and 16 GiB RAM + 16 GiB GPU-Memory.

    To choose the optimal image for your notebooks, you should always try the cheapest and go to the m5.large only if the prior didn't work. You should only choose g4dn if you need a GPU (marked in the Tutorials). It is very expensive and after you exhausted your budget, you cant use your AWS account any more. So watch out and save cost!

- Changing an instance can be done by clicking on x vCPU + y GiB in the top of your notebook and selecting one of the mentioned instance types. Afterwards click safe and continue to initiated the instance switch. This will take a few minutes. 

![](docs/switch-instance.png)

- One more thing you need to know is how to stop a running instance. For example, after you switched the instance you will need to stop the other one in order to save costs. Additionally, mother earth deserves it! To do this, click on the resource section on the left pane in Sagemaker Studio and find the instance in the first section "Running instances". Click the on/off button next to the instance name to stop an instance and accept the pop-up with "Shutdown all". In case you clicked accidentally, you can restart it afterwards again after some time.

![](docs/shutdown-instance.png)



### 7. Optional references


- [Video Getting started with AWS Sagemaker](https://youtu.be/wiDHCWVrjCU)
- [Sagemaker UI overview](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-ui.html)
- [How to use the launcher in Sagemaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-launcher.html)
- [Common tasks in Sagemaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tasks.html)
- [AWS CLI command reference](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/index.html)
- [Git tutorial](https://git-scm.com/) 


## Installing the auto-shutdown extension to save costs


This section is optional, but it helps your team to save costs and helps the environment by reducing carbon emissions! Therefore, we strongly encourage you to go through these steps.

The [auto-shutdown extension](https://github.com/aws-samples/sagemaker-studio-auto-shutdown-extension#sagemaker-studio-autoshutdown-extension) is an extension to Sagemaker Studio which, as the name suggests, automatically stops idle instances after a predefined time. Idle means that there aren't any user initiated activities running on the machine. In this section we will:

+ install the extension to our Sagemaker Studio, 
+ activate the extension, and
+ set a time interval after which the instances are shut down by the extension. 

Again this setup has to be done only once per team.

Let's start by opening a <b>new system terminal</b> from the launcher (refer to 7th step in the previous section to see how this is done). Afterwards copy and paste the following commands:

✋  Attention: please make sure to use a <b> system </b> terminal and not an image terminal. Otherwise this step won't work for you.

# TODO: replace this latest version (this wont work for JupyterLab 3.0 with is GA now)
- Clone the git repository and navigate to the installation script
```bash
git clone https://github.com/steffenkk/sagemaker-studio-lifecycle-config-examples.git sm-extension
cd sm-extension/scripts/install-autoshutdown-server-extension/
```
- Run the script 

```bash 
bash on-jupyter-server-start.sh
```

- After it finished, the terminal will shutdown. This happens, since the jupyter server needs to be restarted. To finish the installation, we need to run one more command. Therefore, reopen the System terminal and run:

```bash
/home/sagemaker-user/.auto-shutdown/set-time-interval.sh
``` 

- Your terminal should tell you now that the auto-shutdown extension has been enabled. Like this: 

![](docs/extension-confirmation.png "Confirmation that the extension is enabled.")

Now the auto shutdown extension is enabled and the time interval is set. Your instances will shut down automatically after 2 hours of idle 
time . You did a great job saving costs and carbon emissions. Planet earth says thanks 🌎 🙏.



## Summary

In this tutorial, we have set up all the necessary resources to participate in the GDSC. Now, your team is ready to take part in this exiting challenge. We hope that everything in this tutorial went fine and that you will have a lots of fun in the course of the challenge. If anything didn't work out as expected, we recommend one last time taking a look at the [Video](https://web.microsoftstream.com/video/ce2af85f-e8b3-4445-ba43-2331baddb64a) for this tutorial. 

All right, thats it! ✅ You can now jump to Tutorial 2: Exploratory Data Analysis. You fill find it in the notebooks folder in the repository we downloaded to your Sagemaker Studio filesystem. Have fun!
