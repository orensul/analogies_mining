package qfirst.frames.annotation

import scala.concurrent.Future

import io.circe.parser.decode
import io.circe.syntax._

case class ClauseAnnotationClient(apiUrl: String) extends ClauseAnnotationService[Future] {

  import scala.concurrent.ExecutionContext.Implicits.global
  val printer = io.circe.Printer.noSpaces

  def getResolution(isFull: Boolean, index: Int): Future[ClauseResolution] = {
    val route = apiUrl + "/" + (if(isFull) "full" else "local") + "/" + index
    org.scalajs.dom.ext.Ajax.get(url = route).map(_.responseText).flatMap { jsonStr =>
      decode[ClauseResolution](jsonStr) match {
        case Left(err)  => Future.failed[ClauseResolution](new RuntimeException(err))
        case Right(res) => Future.successful(res)
      }
    }
  }

  def saveResolution(isFull: Boolean, index: Int, choice: Set[ClauseChoice]): Future[ClauseResolution] = {
    val route = apiUrl + "/" + (if(isFull) "full" else "local") + "/save/" + index
    org.scalajs.dom.ext.Ajax.post(url = route, data = printer.pretty(choice.asJson)).map(_.responseText).flatMap { jsonStr =>
      decode[ClauseResolution](jsonStr) match {
        case Left(err)  => Future.failed[ClauseResolution](new RuntimeException(err))
        case Right(res) => Future.successful(res)
      }
    }
  }
}
