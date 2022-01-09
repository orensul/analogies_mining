package qfirst.clause.ext

import jjm.LowerCaseString
import jjm.implicits._

import io.circe.{Encoder, Decoder}

sealed trait Tense
object Tense {

  sealed trait NonFinite extends Tense {
    import NonFinite._
    override def toString = this match {
      case Bare     => "bare"
      case To       => "to"
      case Gerund   => "gerund"
    }
  }

  object NonFinite {
    case object Bare extends NonFinite
    case object To extends NonFinite
    case object Gerund extends NonFinite

    def fromString(s: String): Option[NonFinite] = s match {
      case "bare"    => Some(Bare)
      case "to"      => Some(To)
      case "gerund"  => Some(Gerund)
      case _ => None
    }

    implicit val nonFiniteTenseEncoder: Encoder[NonFinite] =
      Encoder.encodeString.contramap[NonFinite](_.toString)
    implicit val nonFiniteTenseDecoder: Decoder[NonFinite] =
      Decoder.decodeString.emap(str =>
        fromString(str).toRight(s"Not a valid non-finite tense: $str")
      )
  }

  sealed trait Finite extends Tense {
    import Finite._
    override def toString = this match {
      case Past     => "past"
      case Present  => "present"
      case Modal(m) => m.toString
    }
  }

  object Finite {
    case object Past extends Finite
    case object Present extends Finite
    case class Modal(modalVerb: LowerCaseString) extends Finite
    object Modal {
      val modalVerbStrings: Set[String] =
        Set("can", "will", "might", "would", "should")
      val modalVerbs: Set[LowerCaseString] =
        modalVerbStrings.map(_.lowerCase)
    }

    def fromString(s: String): Option[Finite] = s match {
      case "past"    => Some(Past)
      case "present" => Some(Present)
      case m if Modal.modalVerbStrings.contains(m) => Some(Modal(m.lowerCase))
      case _ => None
    }

    implicit val finiteTenseEncoder: Encoder[Finite] =
      Encoder.encodeString.contramap[Finite](_.toString)
    implicit val finiteTenseDecoder: Decoder[Finite] =
      Decoder.decodeString.emap(str =>
        fromString(str).toRight(s"Not a valid finite tense: $str")
      )
  }
  import NonFinite._
  import Finite._

  implicit val tenseEncoder: Encoder[Tense] =
    Encoder.encodeString.contramap[Tense](_.toString)

  implicit val tenseDecoder: Decoder[Tense] =
    NonFinite.nonFiniteTenseDecoder or Finite.finiteTenseDecoder.map(t => t: Tense)
}
