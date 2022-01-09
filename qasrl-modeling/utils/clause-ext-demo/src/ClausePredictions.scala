// package qfirst.clause.ext.demo
// import qfirst.clause.ext._
// 
// import cats.Id
// import cats.implicits._
// 
// import io.circe.{Encoder, Decoder, ACursor, HCursor, DecodingFailure}
// import io.circe.generic.JsonCodec
// 
// import qasrl.labeling.SlotBasedLabel
// 
// import jjm.DependentPair
// import jjm.DependentMap
// import jjm.LowerCaseString
// import jjm.ling.ESpan
// import jjm.ling.en.InflectedForms
// import jjm.ling.en.VerbForm
// import jjm.implicits._
// 
// // not thread-safe, idgaf
// object VerbInfoGetter {
// 
//   private[this] val cache = collection.mutable.Map.empty[LowerCaseString, (Boolean, TAN)]
// 
//   def getVerbInfo(verbString: LowerCaseString): Either[String, (Boolean, TAN)] = {
//     cache.get(verbString) match {
//       case Some(res) => Right(res)
//       case None =>
//         val stateMachine = new VerbTemplateStateMachine(
//           InflectedForms.generic, false, VerbTemplateStateMachine.TemplateComplete
//         )
//         val verbProcessor = new VerbStringProcessor(stateMachine)
//         for {
//           resultStates <- verbProcessor.processStringFully(verbString.toString).left.map(_ =>
//             s"Invalid verb string: $verbString"
//           ).right
//           resultCompleteStates = resultStates.map(
//             VerbStringProcessor.ValidState.eitherIso.get
//           ).toList.separate._2
//           result <- resultCompleteStates.headOption match {
//             case None => Left(s"Incomplete verb string: $verbString")
//             case Some(x) => Right(x)
//           }
//         } yield {
//           val res = (result.isPassive, result.tan)
//           cache.put(verbString, res)
//           res
//         }
//     }
//   }
// }
// 
// // not thread-safe, idgaf
// object QuestionSlotsGetter {
//   private[this] val cache = collection.mutable.Map.empty[ClausalQuestion, SlotBasedLabel[VerbForm]]
// 
//   def getSlots(clausalQuestion: ClausalQuestion): Either[String, SlotBasedLabel[VerbForm]] = {
//     cache.get(clausalQuestion) match {
//       case Some(res) => Right(res)
//       case None =>
//         val frame = Frame(clausalQuestion.structure, InflectedForms.generic, clausalQuestion.tan)
//         val questionString = frame.questionsForSlot(clausalQuestion.answerSlot).head
//         SlotBasedLabel.getVerbTenseAbstractedSlotsForQuestion(
//           Vector(), InflectedForms.generic, List(questionString)
//         ).head match {
//           case None => Left(s"[Question Error: Conversion] No slots for question: $questionString")
//           case Some(slots) =>
//             cache.put(clausalQuestion, slots)
//             Right(slots)
//         }
//     }
//   }
// }
// 
// case class ClausalQuestion(
//   structure: ArgStructure,
//   tan: TAN,
//   answerSlot: ArgumentSlot
// )
// object ClausalQuestion {
// 
//   private def getAnswerSlotFromString(str: LowerCaseString): ArgumentSlot = str.toString match {
//     case "subj" => Subj
//     case "obj" => Obj
//     case "prep1-obj" => Prep1
//     case "prep2-obj" => Prep2
//     case "misc" => Misc
//     case wh => Adv(wh.lowerCase) // should always be a good wh; TODO maybe check or something
//   }
// 
//   implicit val clausalQuestionEncoder: Encoder[ClausalQuestion] = new Encoder[ClausalQuestion] {
//     def apply(clausalQuestion: ClausalQuestion) = {
//       import io.circe.Json
//       import clausalQuestion._
//       val frame = Frame(structure, InflectedForms.generic, tan)
//       Json.obj(
//         (FrameDataWriter.FrameInfo.getFrameObj(frame).asObject.get.toMap.map {
//            case (k, v) => s"clause-$k" -> v
//          } + ("clause-qarg" -> Json.fromString(FrameDataWriter.FrameInfo.getAnswerSlotLabel(answerSlot)))
//         ).toSeq: _*
//       )
//     }
//   }
// 
//   val readOptionalSlotString = (s: String) => if (s == "_") None else Some(s.lowerCase)
// 
//   def decode(c: ACursor): Decoder.Result[Either[String, ClausalQuestion]] = {
//     for {
//       subj              <- c.downField(     "clause-subj").as[String].right.map(_.lowerCase).right
//       aux               <- c.downField(      "clause-aux").as[String].right.map(readOptionalSlotString).right
//       verbString        <- c.downField(     "clause-verb").as[String].right
//       // verbPrefixAndForm <- verbInfoFromString(verbString).right
//       obj               <- c.downField(      "clause-obj").as[String].right.map(readOptionalSlotString).right
//       prep1             <- c.downField(    "clause-prep1").as[String].right.map(readOptionalSlotString).right
//       prep1Obj          <- c.downField("clause-prep1-obj").as[String].right.map(readOptionalSlotString).right
//       prep2             <- c.downField(    "clause-prep2").as[String].right.map(readOptionalSlotString).right
//       prep2Obj          <- c.downField("clause-prep2-obj").as[String].right.map(readOptionalSlotString).right
//       misc              <- c.downField(     "clause-misc").as[String].right.map(readOptionalSlotString).right
//       answerSlotStr     <- c.downField(     "clause-qarg").as[String].right.map(_.lowerCase).right
//     } yield {
//       def pair[A](slot: ArgumentSlot.Aux[A], value: A): DependentPair[ArgumentSlot.Aux, Id] =
//         DependentPair[ArgumentSlot.Aux, Id, A](slot, value)
//       val fullVerbString = aux.fold(verbString)(_.toString + " " + verbString).lowerCase
//       for {
//         verbInfo <- VerbInfoGetter.getVerbInfo(fullVerbString)
//         prep1Arg <- prep1 match {
//           case None =>
//             if(prep1Obj.nonEmpty) Left("Prep1 has an object but no preposition")
//             else Right(None)
//           case Some(prep) =>
//             Right(Some(Preposition(prep, prep1Obj.map(po => NounLikeArgument.fromPlaceholder(po).get))))
//         }
//         prep2Arg <- prep2 match {
//           case None =>
//             if(prep2Obj.nonEmpty) Left("Prep2 has an object but no preposition")
//             else Right(None)
//           case Some(prep) =>
//             Right(Some(Preposition(prep, prep2Obj.map(po => NounLikeArgument.fromPlaceholder(po).get))))
//         }
//       } yield {
//         val argPairs = List(
//           Some(pair(Subj, Noun.fromPlaceholder(subj).get)),
//           obj.map(o => pair(Obj, Noun.fromPlaceholder(o).get)),
//           prep1Arg.map(p => pair(Prep1, p)),
//           prep2Arg.map(p => pair(Prep2, p)),
//           misc.map(m => pair(Misc, NonPrepArgument.fromPlaceholder(m).get))
//         ).flatten
//         val args = argPairs.foldLeft(DependentMap.empty[ArgumentSlot.Aux, Id])(_ put _)
//         val answerSlot = getAnswerSlotFromString(answerSlotStr)
//         ClausalQuestion(ArgStructure(args, verbInfo._1), verbInfo._2, answerSlot)
//       }
//     }
//   }
// }
// 
// case class ClausalQuestionPrediction(
//   questionSlots: ClausalQuestion,
//   questionProb: Double,
//   invalidProb: Double,
//   answerSpans: List[(ESpan, Double)]
// ) {
//   // def toQuestionPrediction: Either[String, QuestionPrediction] =
//   //   QuestionSlotsGetter.getSlots(questionSlots).map { slots =>
//   //     QuestionPrediction(slots, questionProb, invalidProb, answerSpans)
//   //   }
// }
// object ClausalQuestionPrediction {
//   implicit val clausalQuestionPredictionEncoder: Encoder[ClausalQuestionPrediction] = {
//     import io.circe.generic.semiauto._
//     deriveEncoder[ClausalQuestionPrediction]
//   }
//   def decode(c: ACursor): Decoder.Result[Either[String, ClausalQuestionPrediction]] = {
//     for {
//       clausalQuestionEith <- ClausalQuestion.decode(c.downField("questionSlots"))
//       questionProb        <- c.downField("questionProb").as[Double].right
//       invalidProb         <- c.downField( "invalidProb").as[Double].right
//       answerSpans         <- c.downField( "answerSpans").as[List[(ESpan, Double)]].right
//     } yield {
//       clausalQuestionEith.map(clausalSlots =>
//         ClausalQuestionPrediction(clausalSlots, questionProb, invalidProb, answerSpans)
//       )
//     }
//   }
// }

// case class ClausalVerbPrediction(
//   verbIndex: Int,
//   verbInflectedForms: InflectedForms,
//   questions: List[ClausalQuestionPrediction]
// ) {
//   // def toVerbPrediction = {
//   //   val (qPredErrors, questionPredictions) = questions.map(_.toQuestionPrediction).separate
//   //   qPredErrors.foreach(println)
//   //   VerbPrediction(verbIndex, verbInflectedForms, questionPredictions)
//   // }
// }
// object ClausalVerbPrediction {
//   implicit val clausalVerbPredictionEncoder: Encoder[ClausalVerbPrediction] = {
//     import io.circe.generic.semiauto._
//     deriveEncoder[ClausalVerbPrediction]
//   }
// 
//   implicit val clausalVerbPredictionDecoder: Decoder[ClausalVerbPrediction] = new Decoder[ClausalVerbPrediction] {
//     def apply(c: HCursor): Decoder.Result[ClausalVerbPrediction] = for {
//       verbIndex <- c.downField("verbIndex").as[Int].right
//       verbInflectedForms <- c.downField("verbInflectedForms").as[InflectedForms].right
//       questionsJsons <- c.downField("questions").values match {
//         case None => Left(io.circe.DecodingFailure("Questions are not an array", c.history))
//         case Some(jsons) => Right(jsons)
//       }
//       questionsEithers <- questionsJsons.toList.traverse(
//         q => ClausalQuestionPrediction.decode(q.hcursor)
//       )
//     } yield {
//       val (questionErrors, questions) = questionsEithers.separate
//       questionErrors.map(e => s"[Question Error] $e").foreach(println)
//       ClausalVerbPrediction(verbIndex, verbInflectedForms, questions.toList)
//     }
//   }
// }
// 
// @JsonCodec case class ClausalSentencePrediction(
//   sentenceId: String,
//   sentenceTokens: Vector[String],
//   verbs: List[ClausalVerbPrediction]
// ) {
//   // def toSentencePrediction = SentencePrediction(
//   //   sentenceId, sentenceTokens, verbs.map(_.toVerbPrediction)
//   // )
// }
// object ClausalSentencePrediction
