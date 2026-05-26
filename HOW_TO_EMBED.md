## 嵌入asciicast到README

方案1: 上传到asciinema.org（需注册）
  asciinema upload demo.cast
  → 得到 https://asciinema.org/a/xxxxx 链接
  → README里: [![asciicast](https://asciinema.org/a/xxxxx.svg)](https://asciinema.org/a/xxxxx)

方案2: 自托管（当前方式）
  .cast文件已在repo里，6.9KB
  GitHub不支持直接播放，但任何人clone后:
  asciinema play demo.cast
  
方案3: js播放器
  在README里加:
  <script src="https://asciinema.org/a/xxxxx.js" id="asciicast-xxxxx" async></script>
  需要先上传到asciinema.org
