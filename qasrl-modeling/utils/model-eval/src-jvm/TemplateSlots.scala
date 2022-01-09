package qfirst.model.eval

import qasrl.labeling.SlotBasedLabel

import jjm.LowerCaseString
import jjm.ling.en.VerbForm
import jjm.ling.en.VerbForm.PastParticiple
import jjm.implicits._

case class TemplateSlots(
  wh: LowerCaseString,
  hasSubj: Boolean,
  isPassive: Boolean,
  hasObj: Boolean,
  prep: Option[LowerCaseString],
  obj2: Option[LowerCaseString]
) {
  def toTemplateString = List(
    Some(wh),
    Option("something".lowerCase).filter(_ => hasSubj),
    Some(if(isPassive) "verb[pss]" else "verb").map(_.lowerCase),
    Option("something".lowerCase).filter(_ => hasObj),
    prep,
    obj2
  ).flatten.mkString(" ")
}
object TemplateSlots {
  def fromQuestionSlots(slots: SlotBasedLabel[VerbForm]) = TemplateSlots(
    wh = if(slots.wh.toString == "who") "what".lowerCase else slots.wh,
    hasSubj = slots.subj.nonEmpty,
    isPassive = slots.verb == PastParticiple &&
      (slots.aux.toList ++ slots.verbPrefix).map(_.toString).toSet.intersect(
        Set("be", "been", "is", "isn't", "was", "wasn't")
      ).nonEmpty,
    hasObj = slots.obj.nonEmpty,
    prep = slots.prep,
    obj2 = slots.obj2.map(_.toString.replaceAll("someone", "something").lowerCase)
  )
}
