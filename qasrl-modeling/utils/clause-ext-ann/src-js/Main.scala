// package qfirst.frames.annotation
// 
// import org.scalajs.dom
// 
// import scalacss.DevDefaults._
// 
// import qasrl.bank._
// import qasrl.bank.service._
// 
// import scala.concurrent.Future
// 
// import jjm.ui._
// 
// import nlpdata.util.LowerCaseStrings._
// 
// object Main {
//   def main(args: Array[String]): Unit = {
//     FrameAnnStyles.addToDocument()
//     val docApiEndpoint: String = dom.document
//       .getElementById(SharedConstants.docApiUrlElementId)
//       .getAttribute("value")
//     val annApiEndpoint: String = dom.document
//       .getElementById(SharedConstants.annApiUrlElementId)
//       .getAttribute("value")
//
//     // TODO: don't need the full functionality of doc service. can switch out for simpler service later?
//     import qasrl.bank.service.WebClientDocumentService
//     val dataService = new WebClientDocumentService(docApiEndpoint)
//     object CachedDataService extends DocumentService[CacheCall] {
//       import scala.concurrent.ExecutionContext.Implicits.global
//       import scala.collection.mutable
//       import DocumentService._
//       val documentCache = mutable.Map.empty[DocumentId, Document]
//       val documentRequestCache = mutable.Map.empty[DocumentId, Future[Document]]

//       def getDataIndex = ???

//       def getDocument(id: DocumentId) = {
//         documentCache.get(id).map(Cached(_)).getOrElse {
//           documentRequestCache.get(id).map(Remote(_)).getOrElse {
//             val fut = dataService.getDocument(id)
//             documentRequestCache.put(id, fut)
//             fut.foreach { doc =>
//               documentRequestCache.remove(id)
//               documentCache.put(id, doc)
//             }
//             Remote(fut)
//           }
//         }
//       }

//       def searchDocuments(query: Search.Query) = {
//         ???
//         // if(query.isEmpty) {
//         //   Cached(dataIndex.allDocumentIds)
//         // } else {
//         //   Remote(dataService.searchDocuments(query))
//         // }
//       }
//     }

//     FrameAnnClient.Component(
//       FrameAnnClient.Props(
//         CachedDataService, ClauseAnnotationClient(annApiEndpoint)
//       )
//     ).renderIntoDOM(
//       dom.document.getElementById(SharedConstants.mainDivElementId)
//     )
//   }
// }
