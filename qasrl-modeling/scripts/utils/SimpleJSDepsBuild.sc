import mill._
import ammonite.ops._

trait SimpleJSDeps extends Module {
  def jsDeps = T { Agg.empty[String] }
  def downloadedJSDeps = T {
    for(url <- jsDeps()) yield {
      val filename = url.substring(url.lastIndexOf("/") + 1)
        %("curl", "-o", filename, url)(T.ctx().dest)
      T.ctx().dest / filename
    }
  }
  def aggregatedJSDeps = T {
    val targetPath = T.ctx().dest / "jsdeps.js"
    write.append(targetPath, "")
    downloadedJSDeps().foreach { path =>
      write.append(targetPath, read!(path))
      write.append(targetPath, "\n")
    }
    PathRef(targetPath)
  }
}
