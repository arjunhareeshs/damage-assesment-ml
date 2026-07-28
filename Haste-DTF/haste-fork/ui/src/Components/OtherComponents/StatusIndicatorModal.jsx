// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import React, { useEffect, useRef } from "react";
import SectionModal from "../SectionModal";
import proptypes from "prop-types";

const StatusIndicatorModal = ({ statusMessages, infoMetadata, onClose }) => {
  const modalBodyRef = useRef(null);

  StatusIndicatorModal.propTypes = {
    statusMessages: proptypes.array.isRequired,
    // Optional run-parameter rows rendered above the status-message table.
    // Each item is {label, value}; consumers (e.g. EmbeddingModelRow) build
    // this from the saved Model record.
    infoMetadata: proptypes.arrayOf(
      proptypes.shape({
        label: proptypes.string.isRequired,
        value: proptypes.node.isRequired,
      })
    ),
    onClose: proptypes.func.isRequired,
  };

  // Scroll to the bottom when statusMessages updates
  useEffect(() => {
    if (modalBodyRef.current) {
      modalBodyRef.current.scrollTop = modalBodyRef.current.scrollHeight;
    }
  }, [statusMessages]);

  if (statusMessages.length === 0 && (!infoMetadata || infoMetadata.length === 0)) {
    return null;
  }

  return (
    <SectionModal
      title={"Status Messages"}
      body={
        <>
          {infoMetadata && infoMetadata.length > 0 && (
            <div className="row mb-3">
              <div className="col-12">
                <table>
                  <tbody>
                    {infoMetadata.map((item, index) => (
                      <tr key={`meta-${index}`}>
                        <td className="pe-3 fw-semibold">{item.label}</td>
                        <td>{item.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          <div
            className="row mb-2"
            style={{ maxHeight: "300px", overflowY: "auto" }}
            ref={modalBodyRef}
          >
            <div className="col-12 d-flex flex-column">
              <table>
                <thead>
                  <tr key="header">
                    <td>
                      <b>Timestamp</b>
                    </td>
                    <td>
                      <b>Message</b>
                    </td>
                  </tr>
                </thead>
                <tbody>
                  {statusMessages.map((message, index) => {
                    return (
                      <tr key={index}>
                        <td className="pe-3">{message.timestamp}</td>
                        <td>{message.message}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      }
      onClose={onClose}
      icon="Info"
    />
  );
};

export default StatusIndicatorModal;
