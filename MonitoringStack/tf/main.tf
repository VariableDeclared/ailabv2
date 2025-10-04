resource "lxd_instance" "k8s" {
  name  = "k8s-stack"
  image = "ubuntu-daily:24.04"
  type = "virtual-machine"

  limits = {
    cpu = 2
    memory = "4GiB"
  }
    # Cloud-init user-data for customization
  config = {
    "boot.autostart" = true
    "user.user-data" = <<-EOT
      #cloud-config
      # hostname: tf-cloudinit
      timezone: UTC

      package_update: true
      package_upgrade: true
      packages:
        - htop
        - curl
        - git
        - zsh
        - containerd
        - apt-transport-https 
        - ca-certificates 
        - gpg

      users:
        - name: devuser
          groups: sudo
          shell: /bin/bash
          sudo: ALL=(ALL) NOPASSWD:ALL
          ssh_authorized_keys:
            - ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBu8meueQA8idgOrVV83hOAJBgMzwrdmAP2m6Vb3otA1

      runcmd:
        - sudo swapoff -a
        - curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.33/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
        - echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.33/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
        - sudo apt-get update
        - sudo apt-get install -y kubelet kubeadm kubectl
        - sudo apt-mark hold kubelet kubeadm kubectl
        - sudo systemctl enable --now kubelet

    EOT
  }
}