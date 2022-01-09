package qfirst.clause.ext

import cats.Id
import cats.effect.IO

import io.circe.Json
import io.circe.{Encoder, Decoder}
import io.circe.HCursor

import jjm.DependentMap
import jjm.LowerCaseString
import jjm.ling.en.InflectedForms
import jjm.implicits._

import qasrl.data.Dataset

/**
 * Used previously to write resolved-clause data to the slot-based format,
 * when I was still using the elaborated clause format in the ML models.
 * Now I'm instead doing a simpler resolution process that has fewer slots
 * and reuses the old QA-SRL Frame object; see core/src/ClauseResolution.scala
 * and core/src-jvm/ClauseResolutionApp.scala
 *
 * However, I'm still keeping around this clause stuff because I'm using it
 * to demo the clausal structure (though it's a to-do to change that over) and
 * I might expand it in the future to proper subcategorization frames, in which
 * case this might be a useful reference.
 */
object ClauseDataWriter {
  def recapitalizeInflection(s: String): String = s match {
    case "presentsingular3rd" => "presentSingular3rd"
    case "presentparticiple" => "presentParticiple"
    case "pastparticiple" => "pastParticiple"
    case x => x
  }

  case class ClauseInfo(
    sentenceId: String,
    verbIndex: Int,
    question: String,
    frame: Frame, // DependentMap[ArgumentSlot.Aux, Id],
    answerSlot: ArgumentSlot)
  object ClauseInfo {
    def getSlotLabel(slot: ArgumentSlot): String = slot match {
      case Subj => "subj"
      case Obj => "obj"
      case Prep1 => "prep1"
      case Prep2 => "prep2"
      case Misc => "misc"
      case Adv(wh) => wh.toString
    }
    def getAnswerSlotLabel(slot: ArgumentSlot): String = slot match {
      case Prep1 => "prep1-obj"
      case Prep2 => "prep2-obj"
      case x => getSlotLabel(x)
    }
    def getAnswerSlotFromLabel(label: String): ArgumentSlot = label match {
      case "prep1-obj" | "prep1" => Prep1
      case "prep2-obj" | "prep2" => Prep2
      case "subj" => Subj
      case "obj" => Obj
      case "misc" => Misc
      case wh => Adv(wh.lowerCase)
    }

    def getFrameObj(frame: Frame) = {
      val verbTokens = frame.copy(verbInflectedForms = InflectedForms.generic).getVerbStack.map(recapitalizeInflection)
      val (auxSlotValue, verbSlotValue) = verbTokens.tail match {
        case Nil => ("_", verbTokens.toList.mkString(" "))
        case toks => (verbTokens.head, toks.mkString(" "))
      }
      Json.obj(
        (List(
           getSlotLabel(Subj) -> Json.fromString(frame.args.get(Subj).fold("_")(_.placeholder.mkString(" "))),
           getSlotLabel(Obj)  -> Json.fromString(frame.args.get(Obj).fold("_")(_.placeholder.mkString(" "))),
           getSlotLabel(Misc)  -> Json.fromString(frame.args.get(Misc).fold("_")(_.placeholder.mkString(" "))),
           "aux" -> Json.fromString(auxSlotValue),
           "verb" -> Json.fromString(verbSlotValue),
           ) ++ List(Prep1, Prep2).flatMap(prepSlot =>
           List(
             getSlotLabel(prepSlot) -> Json.fromString(
               frame.args.get(prepSlot).fold("_")(_.preposition.toString)
             ),
             (getAnswerSlotLabel(prepSlot)) -> Json.fromString(
               frame.args.get(prepSlot).flatMap(_.objOpt).fold("_")(_.placeholder.mkString(" "))
             )
           )
         )): _*
      )
    }

    import io.circe.syntax._

    implicit val clauseInfoEncoder: Encoder[ClauseInfo] = new Encoder[ClauseInfo] {
      final def apply(info: ClauseInfo): Json = Json.obj(
        "sentenceId" -> Json.fromString(info.sentenceId),
        "verbIndex" -> Json.fromInt(info.verbIndex),
        "question" -> Json.fromString(info.question),
        "frame" -> info.frame.asJson,
        "answerSlot" -> Json.fromString(getAnswerSlotLabel(info.answerSlot)),
        "slots" -> getFrameObj(info.frame),
      )
    }

    implicit val clauseInfoDecoder: Decoder[ClauseInfo] = new Decoder[ClauseInfo] {
      final def apply(c: HCursor): Decoder.Result[ClauseInfo] = for {
        sentenceId <- c.get[String]("sentenceId")
        verbIndex <- c.get[Int]("verbIndex")
        question <- c.get[String]("question")
        frame <- c.get[Frame]("frame")
        answerSlot <- c.get[String]("answerSlot").map(getAnswerSlotFromLabel)
      } yield ClauseInfo(sentenceId, verbIndex, question, frame, answerSlot)
    }
  }
}
