terraform {
  required_providers {
    lxd = {
      source = "terraform-lxd/lxd"
      version = "2.5.0"
    }
  }
}

provider "lxd" {
  # Configuration options
}