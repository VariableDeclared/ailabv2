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
          ssh_import_id:
            - lp:pjds
      write_files:
      - content: |
          curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.33/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
          echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.33/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
          sudo apt-get update
          sudo apt-get install -y kubelet kubeadm kubectl
          sudo apt-mark hold kubelet kubeadm kubectl
          sudo systemctl enable --now kubelet
          kubeadm init --skip-phases=addon/kube-proxy
          CILIUM_CLI_VERSION=$(curl -s https://raw.githubusercontent.com/cilium/cilium-cli/main/stable.txt)
          CLI_ARCH=amd64
          if [ "$(uname -m)" = "aarch64" ]; then CLI_ARCH=arm64; fi
          curl -L --fail --remote-name-all https://github.com/cilium/cilium-cli/releases/download/${CILIUM_CLI_VERSION}/cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}
          sha256sum --check cilium-linux-${CLI_ARCH}.tar.gz.sha256sum
          sudo tar xzvfC cilium-linux-${CLI_ARCH}.tar.gz /usr/local/bin
          rm cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}
          cilium install --version 1.18.2
        path: /opt/k8s-init.sh
        permissions: 0700
        owner: root:root
      write_files:
      - content: |
          mkdir -p $HOME/.kube
          sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
          sudo chown $(id -u):$(id -g) $HOME/.kube/config
        path: /opt/instructions.txt
        permissions: 0600
        owner: root:root

      runcmd:
        - sudo swapoff -a
        - /opt/k8s-init.sh
EOT
  }
}