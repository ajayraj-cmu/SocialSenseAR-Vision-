/**
 * Terraform Variables
 */

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "instance_type" {
  description = "EC2 instance type (GPU instance)"
  type        = string
  default     = "g4dn.xlarge"
}

variable "ami_id" {
  description = "AMI ID for EC2 instances (Deep Learning AMI with GPU support)"
  type        = string
  # This should be updated based on your region
  # Get latest with: aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*"
  default = ""
}

variable "min_instances" {
  description = "Minimum number of instances in Auto Scaling Group"
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum number of instances in Auto Scaling Group"
  type        = number
  default     = 10
}

variable "desired_instances" {
  description = "Desired number of instances in Auto Scaling Group"
  type        = number
  default     = 2
}
