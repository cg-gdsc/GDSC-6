## A list of CLI commands you need when you are only going through the video

1. Commands to run 

- 👋  Attention: replace ROLENAME with your role's name

```bash
git clone https://gdsc-code-commit-user-at-954362353459:6intZSiLZxXrdboGMDDSvf9VvnpiPOo+0JoiYxZBSq4=@git-codecommit.eu-central-1.amazonaws.com/v1/repos/gdsc5-tutorials-public

cd gdsc5-tutorials-public

aws iam put-role-policy --role-name ROLENAME --policy-name sm-execution-policy --policy-document file://scripts/create-domain/inline-policy.json 
```


2. Image URI: 

```
954362353459.dkr.ecr.us-east-1.amazonaws.com/sm-training-custom:torch-1.8.1-cu111
```