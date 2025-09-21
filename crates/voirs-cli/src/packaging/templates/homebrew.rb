class {{NAME}} < Formula
  desc "{{DESCRIPTION}}"
  homepage "{{HOMEPAGE}}"
  url "{{REPOSITORY}}/releases/download/v{{VERSION}}/voirs-macos.tar.gz"
  sha256 "{{SHA256}}"
  license "{{LICENSE}}"
  version "{{VERSION}}"

  depends_on "ffmpeg" => :optional

  def install
    bin.install "voirs"
  end

  test do
    system "#{bin}/voirs", "--version"
  end
end